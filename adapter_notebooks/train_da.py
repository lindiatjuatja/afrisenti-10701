def run_da_experiment(args, encode_batch, parallel, train_lm, src_lm, tgt_lm):
    import pandas as pd
    import torch
    import numpy as np
    from sklearn.metrics import f1_score, balanced_accuracy_score
    from tqdm import tqdm
    import os

    from pytorch_adapt.containers import Models
    from pytorch_adapt.models import Discriminator, Classifier
    from torch import nn
    import gc
    import copy
    from utils import get_source_data, get_target_data, make_model
    from transformers import (
        AutoModelForMaskedLM,
        AutoAdapterModel,
    )
    from transformers.adapters.configuration import AdapterConfig
    from transformers.adapters.composition import Stack, Parallel


    import warnings
    warnings.filterwarnings(action='ignore')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    def adapt_encode(row, src=0):
        out = encode_batch(row)
        if parallel:
            if src==1:
                return {
                    'imgs': torch.vstack([out['input_ids'], out['attention_mask'], torch.ones_like(out['attention_mask'])]),
                    'labels': torch.LongTensor([out['labels']])[0],
                    'domain': torch.LongTensor([row.domain])[0]
                }
            else:
                return {
                    'imgs': torch.vstack([out['input_ids'], out['attention_mask']]),
                    'labels': torch.LongTensor([out['labels']])[0],
                    'domain': torch.LongTensor([row.domain])[0]
                }
        return {
            'imgs': torch.vstack([out['input_ids'], out['attention_mask']]),
            'labels': torch.LongTensor([out['labels']])[0],
            'domain': torch.LongTensor([row.domain])[0]
        }


    def load_target_data(lang_code):
        am_train, am_dev, am_test = get_target_data(args, test=True, lang_code=lang_code)
        for am in [am_train, am_dev, am_test]:
            am['domain'] = 1
        am_train = am_train.apply(adapt_encode, axis=1)
        am_dev = am_dev.apply(adapt_encode, axis=1)
        am_test = am_test.apply(adapt_encode, axis=1)
        return am_train, am_dev, am_test



    en_train, en_dev = get_source_data(args, dev=True)
    en_train['domain'] = 0
    en_dev['domain'] = 0
    am_train, am_dev, am_test = load_target_data(args.lang_code)

    en_train = en_train.apply(lambda x: adapt_encode(x, src=1), axis=1)
    en_dev = en_dev.apply(lambda x: adapt_encode(x, src=1), axis=1)
                            
    print(f'Source train size: {len(en_train)},     dev size: {len(en_dev)}')
    print(f'Target train size: {len(am_train)},     dev size: {len(am_dev)},      test size: {len(am_test)}')

    class SimpleSourceAndTargetDataset(torch.utils.data.Dataset):
        def __init__(self, s, t):
            self.s = s
            self.t = t

        def __len__(self) -> int:
            return len(self.t)

        def __getitem__(self, idx):
            tgt = self.t.iloc[idx]
            src = self.s.iloc[self.get_random_src_idx()]
            return {
                'src_imgs': src['imgs'].to(device),
                'src_labels': src['labels'].to(device),
                'src_domain': src['domain'].to(device),
                'target_imgs': tgt['imgs'].to(device),
                # 'target_labels': tgt['labels'].to(device),
                'target_domain': tgt['domain'].to(device),
            }
        
        def get_random_src_idx(self):
            return np.random.choice(len(self.s))
        
    class SimpleTargetDataset(torch.utils.data.Dataset):
        def __init__(self, t):
            self.t = t

        def __len__(self) -> int:
            return len(self.t)

        def __getitem__(self, idx):
            tgt = self.t.iloc[idx]
            return {
                'target_imgs': tgt['imgs'].to(device),
                'target_labels': tgt['labels'].to(device),
                'target_domain': tgt['domain'].to(device),
            }

    source_train_data = SimpleSourceAndTargetDataset(en_train, am_train)
    source_valid_data = SimpleTargetDataset(en_dev)
    train_data = SimpleTargetDataset(am_train)
    valid_data = SimpleTargetDataset(am_dev)
    test_data = SimpleTargetDataset(am_test)

    class Generator(nn.Module):
        def __init__(self, ):
            super().__init__()

            adapter_config = AdapterConfig.load(args.adapter_type)
            if train_lm and parallel:
                lang_adapter_config = AdapterConfig.load(args.adapter_type, reduction_factor=2)
                self.model = AutoAdapterModel.from_pretrained(args.base_model)
                src_adapter = self.model.load_adapter(args.lm_zero_src_lm_adapter, config=lang_adapter_config)
                tgt_adapter = self.model.load_adapter(args.lm_zero_tgt_lm_adapter, config=lang_adapter_config)
                self.model.add_adapter('sa', config=adapter_config)
                self.model.train_adapter(['sa'])
                self.model.active_adapters = Parallel(
                    Stack(src_adapter, 'sa'), 
                    Stack(tgt_adapter, 'sa'))
            elif parallel:
                self.model = AutoAdapterModel.from_pretrained(args.base_model)
                self.model.add_adapter('sa_src', config=adapter_config)
                self.model.add_adapter('sa_tgt', config=adapter_config)
                self.model.train_adapter(['sa_src', 'sa_tgt'])
                self.model.active_adapters = Parallel('sa_src', 'sa_tgt')
            elif train_lm:
                lang_adapter_config = AdapterConfig.load(args.adapter_type, reduction_factor=2)
                self.model = AutoAdapterModel.from_pretrained(args.base_model)
                src_adapter = self.model.load_adapter(args.lm_zero_src_lm_adapter, config=lang_adapter_config)
                tgt_adapter = self.model.load_adapter(args.lm_zero_tgt_lm_adapter, config=lang_adapter_config)
                self.model.add_adapter('sa', config=adapter_config)
                self.model.train_adapter(['sa'])
                self.model.active_adapters = Stack(tgt_adapter, "sa")
            else:
                self.model = AutoAdapterModel.from_pretrained(args.base_model)
                self.model.add_adapter('sa', config=adapter_config)
                self.model.train_adapter(['sa'])
                self.model.set_active_adapters('sa')
            
            self.project = nn.Sequential(
                nn.Linear(768, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(768, args.da_repr)
            )

        def forward(self, x):
            if parallel:
                if x.shape[1] == 3:
                    a = self.model(x[:, 0], x[:, 1])
                    return self.project(a['last_hidden_state'][:x.shape[0], 0])
                else:
                    a = self.model(x[:, 0], x[:, 1])
                    return self.project(a['last_hidden_state'][x.shape[0]:, 0])
            else:
                a = self.model(x[:, 0], x[:, 1])
                a = self.project(a['last_hidden_state'][:, 0])
                return a

    updates_per_epoch = 4

    source_train_dataloader = torch.utils.data.DataLoader(source_train_data, batch_size=args.per_device_batch_size, shuffle=True)
    source_valid_dataloader = torch.utils.data.DataLoader(source_valid_data, batch_size=args.per_device_batch_size, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.per_device_batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=args.per_device_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.per_device_batch_size, shuffle=False)

    num_source_batches = len(source_train_dataloader)
    num_batches = len(train_dataloader)
    criterion = nn.CrossEntropyLoss()
    
    G = Generator().to(device) 

    C = Classifier(3, in_size=args.da_repr, h=args.da_Ch).to(device)

    # G_opt = torch.optim.AdamW(G.parameters(), lr=args.lr)
    G_opt = torch.optim.Adam([
                {'params': G.model.parameters()},
                {'params': G.project.parameters(), 'lr': args.da_lr}
            ], lr=args.lr)
    C_opt = torch.optim.AdamW(C.parameters(), lr=args.da_lr)

    D = Discriminator(in_size=args.da_repr, h=args.da_Dh).to(device)
    D_opt = torch.optim.AdamW(D.parameters(), lr=args.da_lr)

    misc = dict()

    if args.da_method == 'adda':
        from pytorch_adapt.hooks import ADDAHook
        T = copy.deepcopy(G)
        T_opt = torch.optim.AdamW(T.parameters(), lr=args.lr)
        hook = ADDAHook(g_opts=[T_opt], d_opts=[D_opt])
        models = {"G": G, "C": C, "D": D, 'T':T}

    elif args.da_method == 'coral':
        from pytorch_adapt.hooks import AlignerPlusCHook
        from pytorch_adapt.layers import CORALLoss

        del D
        del D_opt

        hook = AlignerPlusCHook(opts=[G_opt, C_opt], loss_fn=CORALLoss(), softmax=False)
        models = {"G": G, "C": C}

    elif args.da_method == 'cdan':
        from pytorch_adapt.hooks import CDANHook
        from pytorch_adapt.layers import RandomizedDotProduct
        feature_combiner = RandomizedDotProduct(in_dims=[args.da_repr, 3], out_dim=args.da_repr)
        hook = CDANHook(g_opts=[G_opt, C_opt], d_opts=[D_opt])

        models = {"G": G, "C": C, "D": D}
        misc = {"feature_combiner": feature_combiner}

    elif args.da_method == 'dann':
        from pytorch_adapt.hooks import DANNHook
        hook = DANNHook(opts=[G_opt, C_opt, D_opt])
        models = {"G": G, "C": C, "D": D}
    
    elif args.da_method == 'dc':
        from pytorch_adapt.hooks import DomainConfusionHook

        D = Discriminator(in_size=args.da_repr, h=args.da_Dh, out_size=2).to(device)
        D_opt = torch.optim.AdamW(D.parameters(), lr=args.da_lr)

        hook = DomainConfusionHook(g_opts=[G_opt, C_opt], d_opts=[D_opt])
        models = {"G": G, "C": C, "D": D}
    
    elif args.da_method == 'gan':
        from pytorch_adapt.hooks import GANHook
        hook = GANHook(g_opts=[G_opt, C_opt], d_opts=[D_opt])
        models = {"G": G, "C": C, "D": D}

    elif args.da_method == 'itl':
        from pytorch_adapt.hooks import (
            ClassifierHook,
            ISTLossHook,
            TargetDiversityHook,
            TargetEntropyHook,
        )
        del D
        del D_opt
        hook = ClassifierHook(
            opts=[G_opt, C_opt],
            post=[ISTLossHook(), TargetEntropyHook(), TargetDiversityHook()],
        )

        models = {"G": G, "C": C}
    elif args.da_method == 'none':
        models = {"G": G, "C": C}
    else:
        assert False, "DA method not supported"

    models = Models(models)
    gc.collect()

    update_idxs = set([i * (num_source_batches // updates_per_epoch) 
        for i in range(1, updates_per_epoch)] + [num_source_batches])

    best_losses = dict()
    best_valid = -1

    save_path = args.tmp_folder + 'saved_model/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.train_da:
        print("Training DA model")
        for epoch in range(1, 1+args.da_epochs):
            total_loss = 0.0 

            pbar = tqdm(source_train_dataloader, desc=f"Epoch {epoch}", leave=args.show_bar)
            for idx, data in enumerate(pbar, start=1):
                models.train()
                if args.da_method == 'none':
                    
                    G_opt.zero_grad()
                    C_opt.zero_grad()
                    logits = C(G(data['src_imgs']))
                    loss = criterion(logits, data['src_labels'])
                    total_loss += loss.item()
                    loss.backward()
                    G_opt.step()
                    C_opt.step()
                else:
                    _, loss = hook({**models, **misc, **data})
                    if args.da_method in ['dann', 'itl', 'coral']:
                        loss = loss['total_loss']
                    elif args.da_method in ['adda', 'cdan', 'gan', 'dc']:
                        loss = loss['g_loss']
                    total_loss += loss['total']

                if idx in update_idxs:
                    
                    models.eval()
                    with torch.no_grad():
                        logits = []
                        ans = []
                        for data in source_valid_dataloader:
                            logits.append(C(G(data["target_imgs"])))
                            ans.append(data["target_labels"])
                        source_valid_preds = torch.cat(logits, dim=0).argmax(-1).cpu().numpy()
                        source_valid_ans = torch.cat(ans, dim=0).cpu().numpy()
                        source_valid_bal_acc = balanced_accuracy_score(source_valid_preds, source_valid_ans)


                        logits = []
                        ans = []
                        for data in valid_dataloader:
                            logits.append(C(G(data["target_imgs"])))
                            ans.append(data["target_labels"])
                        valid_preds = torch.cat(logits, dim=0).argmax(-1).cpu().numpy()
                        valid_ans = torch.cat(ans, dim=0).cpu().numpy()
                        valid_bal_acc = balanced_accuracy_score(valid_ans, valid_preds)
                        
                        if valid_bal_acc > best_valid:
                            best_valid = valid_bal_acc
                            best_losses = dict()
                            
                            best_losses['dev_balanced_accuracy'] = valid_bal_acc
                            best_losses['dev_f1'] = f1_score(valid_ans, valid_preds, average='weighted')
                            
                            logits = []
                            ans = []
                            for data in test_dataloader:
                                logits.append(C(G(data["target_imgs"])))
                                ans.append(data["target_labels"])
                            test_preds = torch.cat(logits, dim=0).argmax(-1).cpu().numpy()
                            test_ans = torch.cat(ans, dim=0).cpu().numpy()
                            
                            best_losses['test_balanced_accuracy'] = balanced_accuracy_score(test_ans, test_preds)
                            best_losses['test_f1'] = f1_score(test_ans, test_preds, average='weighted')

                            torch.save(G.state_dict(), save_path + 'G')
                            torch.save(C.state_dict(), save_path + 'C')
                            
                            # G.model.save_adapter(args.tmp_folder + 'da/', 'sa')
                        del logits
                        del ans
                    models.train()


                    pbar.set_description(f" Epoch {epoch} | tr {total_loss / idx:.2f} | source valid bal_acc {source_valid_bal_acc:.3f}" + \
                                        f" | valid bal_acc {valid_bal_acc:.3f} | test bal_acc {best_losses['test_balanced_accuracy']:.3f}" + \
                                            f" | test f1 {best_losses['test_f1']:.3f}")
    #                 train_losses.append(total_loss / idx)
        print("\n\nTest results on best validation, zero shot")
        for key, value in best_losses.items():
            print(key, ':', value)

    print('\nloading previous best model')
    G.load_state_dict(torch.load( save_path + 'G'))
    C.load_state_dict(torch.load( save_path + 'C'))
    gc.collect()
    if args.da_test_all:
        print('\n\nTesting on Other Languages')
        models.eval()
        for lang_code in ['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo']:
            print(lang_code, 'Test results:')
            _, _, test = load_target_data(lang_code)
            lang_test_data = SimpleTargetDataset(test) 
            lang_test_dataloader = torch.utils.data.DataLoader(lang_test_data, batch_size=args.per_device_batch_size, shuffle=False)
            logits = []
            ans = []
            with torch.no_grad():
                for data in lang_test_dataloader:
                    logits.append(C(G(data["target_imgs"])))
                    ans.append(data["target_labels"])
            test_preds = torch.cat(logits, dim=0).argmax(-1).cpu().numpy()
            test_ans = torch.cat(ans, dim=0).cpu().numpy()
            print('Balanced Accuracy: ', balanced_accuracy_score(test_ans, test_preds))
            print('F1: ', f1_score(test_ans, test_preds, average='weighted'), '\n')
        models.train()



    if args.da_finetune:
        print('\n\n\n Finetuning')
        G_opt = torch.optim.AdamW(G.parameters(), lr=args.da_finetune_lr)
        C_opt = torch.optim.AdamW(C.parameters(), lr=args.da_finetune_lr)
        update_idxs = set([i * (num_batches // updates_per_epoch) 
            for i in range(1, updates_per_epoch)] + [num_batches])

        best_losses = dict()
        best_valid = -1
        for epoch in range(1, 1+args.da_finetune_epochs):
            total_loss = 0.0 

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=args.show_bar)
            for idx, data in enumerate(pbar, start=1):
                models.train()
                    
                G_opt.zero_grad()
                C_opt.zero_grad()
                logits = C(G(data['target_imgs']))
                loss = criterion(logits, data['target_labels'])
                total_loss += loss.item()
                loss.backward()
                G_opt.step()
                C_opt.step()

                if idx in update_idxs:
                    
                    models.eval()
                    with torch.no_grad():
                        logits = []
                        ans = []
                        for data in valid_dataloader:
                            logits.append(C(G(data["target_imgs"])))
                            ans.append(data["target_labels"])
                        valid_preds = torch.cat(logits, dim=0).argmax(-1).cpu().numpy()
                        valid_ans = torch.cat(ans, dim=0).cpu().numpy()
                        valid_bal_acc = balanced_accuracy_score(valid_ans, valid_preds)
                        
                        if valid_bal_acc > best_valid:
                            best_valid = valid_bal_acc
                            best_losses = dict()
                            
                            best_losses['dev_balanced_accuracy'] = valid_bal_acc
                            best_losses['dev_f1'] = f1_score(valid_ans, valid_preds, average='weighted')
                            
                            logits = []
                            ans = []
                            for data in test_dataloader:
                                logits.append(C(G(data["target_imgs"])))
                                ans.append(data["target_labels"])
                            test_preds = torch.cat(logits, dim=0).argmax(-1).cpu().numpy()
                            test_ans = torch.cat(ans, dim=0).cpu().numpy()
                            
                            best_losses['test_balanced_accuracy'] = balanced_accuracy_score(test_ans, test_preds)
                            best_losses['test_f1'] = f1_score(test_ans, test_preds, average='weighted')
                    models.train()

                    pbar.set_description(f" Epoch {epoch} | tr {total_loss / idx:.2f}" + \
                                        f" | valid bal_acc {valid_bal_acc:.3f} | test bal_acc {best_losses['test_balanced_accuracy']:.3f}" + \
                                            f" | test f1 {best_losses['test_f1']:.3f}")

        print("Test results after tuning")
        for key, value in best_losses.items():
            print(key, ':', value)

    return args.tmp_folder + 'da/'