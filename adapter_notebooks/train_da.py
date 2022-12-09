def run_da_experiment(args, encode_batch, train_lm, src_lm, tgt_lm):
    import pandas as pd
    import torch
    import numpy as np
    from sklearn.metrics import f1_score, balanced_accuracy_score
    from tqdm import tqdm

    from pytorch_adapt.containers import Models
    from pytorch_adapt.models import Discriminator, Classifier
    from torch import nn
    import gc
    import copy
    from utils import get_source_data, get_target_data, make_model


    import warnings
    warnings.filterwarnings(action='ignore')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    en_train = get_source_data(args, dev=False)
    am_train, am_dev, am_test = get_target_data(args, test=True)

    en_train['domain'] = 0
    for am in [am_train, am_dev, am_test]:
        am['domain'] = 1

    def adapt_encode(row, src=0):
        out = encode_batch(row)
        if train_lm:
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

    en_train = en_train.apply(lambda x: adapt_encode(x, src=1), axis=1)
                            
    am_train = am_train.apply(adapt_encode, axis=1)
    am_dev = am_dev.apply(adapt_encode, axis=1)
    am_test = am_test.apply(adapt_encode, axis=1)

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

    train_data = SimpleSourceAndTargetDataset(en_train, am_train)
    valid_data = SimpleTargetDataset(am_dev)
    test_data = SimpleTargetDataset(am_test)

    class Generator(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.model = make_model(args, add_head=False, task_name='da')

        def forward(self, x):
            a = self.model(x[:, 0], x[:, 1]).pooler_output
            return a

    class LMGenerator(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.model = make_model(args, add_head=False, task_name='da', parallel=(src_lm, tgt_lm))
        
        def forward(self, x):
            if x.shape[1] == 3:
                # source
                a = self.model(x[:, 0], x[:, 1])[0].pooler_output
            else:
                # target
                a = self.model(x[:, 0], x[:, 1])[1].pooler_output
            return a

    updates_per_epoch = 4

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.per_device_batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=args.per_device_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.per_device_batch_size, shuffle=False)

    num_batches = len(train_dataloader)
    num_valid_batches = len(valid_dataloader)
    num_test_batches = len(test_dataloader)
    
    if train_lm:
        G = LMGenerator().to(device) 
    else:
        G = Generator().to(device) 

    C = Classifier(3, in_size=768, h=args.da_Ch).to(device)

    G_opt = torch.optim.AdamW(G.parameters(), lr=args.lr)
    C_opt = torch.optim.AdamW(C.parameters(), lr=args.da_lr)

    D = Discriminator(in_size=768, h=args.da_Dh).to(device)
    D_opt = torch.optim.AdamW(D.parameters(), lr=args.da_lr)

    misc = dict()

    if args.da_method == 'adda':
        from pytorch_adapt.hooks import ADDAHook
        T = copy.deepcopy(G)
        T_opt = torch.optim.AdamW(T.parameters(), lr=args.lr)
        hook = ADDAHook(g_opts=[T_opt], d_opts=[D_opt])
        models = Models({"G": G, "C": C, "D": D, 'T':T})

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
        feature_combiner = RandomizedDotProduct(in_dims=[768, 3], out_dim=768)
        hook = CDANHook(g_opts=[G_opt, C_opt], d_opts=[D_opt])

        models = {"G": G, "C": C, "D": D}
        misc = {"feature_combiner": feature_combiner}

    elif args.da_method == 'dann':
        from pytorch_adapt.hooks import DANNHook
        hook = DANNHook(opts=[G_opt, C_opt, D_opt])
        models = {"G": G, "C": C, "D": D}
    
    elif args.da_method == 'dc':
        from pytorch_adapt.hooks import DomainConfusionHook

        D = Discriminator(in_size=768, h=args.da_Dh, out_size=2).to(device)
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
    else:
        assert False, "DA method not supported"

    models = Models(models)
    gc.collect()

    update_idxs = set([i * (num_batches // updates_per_epoch) 
        for i in range(1, updates_per_epoch)] + [num_batches])

    best_losses = dict()
    best_valid = -1

    print("Training DA model")
    for epoch in range(1, 1+args.train_epochs):
        total_loss = 0.0 

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=args.show_bar)
        for idx, data in enumerate(pbar, start=1):
            models.train()
            _, loss = hook({**models, **misc, **data})
            
            total_loss += loss['total_loss']['total']

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
                        
                        G.model.save_adapter(args.tmp_folder + 'da/', 'da')


                pbar.set_description(f" Epoch {epoch} | tr {total_loss / idx:.3f}" + \
                                    f" | valid bal_acc {valid_bal_acc:.2f} | test bal_acc {best_losses['test_balanced_accuracy']:.2f}" + \
                                        f" | test f1 {best_losses['test_f1']:.2f}")
#                 train_losses.append(total_loss / idx)
    print("Test results on best validation, zero shot")
    for key, value in best_losses.items():
        print(key, ':', value)
    return args.tmp_folder + 'da/'