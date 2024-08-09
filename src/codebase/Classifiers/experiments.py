import gc
import time
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .models.breast_clip_classifier import BreastClipClassifier
from Datasets.dataset_utils import get_dataloader_RSNA
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from metrics import pfbeta_binarized, pr_auc, compute_auprc, auroc, compute_accuracy_np_array
from utils import seed_all, AverageMeter, timeSince
from sklearn.metrics import classification_report


def do_experiments(args, device):
    if 'efficientnetv2' in args.arch:
        args.model_base_name = 'efficientv2_s'
    elif 'efficientnet_b5_ns' in args.arch:
        args.model_base_name = 'efficientnetb5'
    else:
        args.model_base_name = args.arch

    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    if args.df[args.label].nunique()==2:
        args.binary = True
    else:
        args.binary = False
    if args.label.lower() == "density" and args.dataset.lower() == "rsna":
        args.df[args.label] = args.df[args.label].map({'A': 0,'B' : 1, 'C' : 2, 'D' : 3})
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)
    oof_df = pd.DataFrame()
    for fold in range(args.start_fold, args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)
        if args.dataset.lower() == "rsna":
            args.train_folds = args.df[
                (args.df['fold'] == 1) | (args.df['fold'] == 2)].reset_index(drop=True)
            args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)
            print(f"train_folds shape: {args.train_folds.shape}")
            print(f"valid_folds shape: {args.valid_folds.shape}")

        elif args.dataset.lower() == "vindr":
            args.train_folds = args.df[args.df['split'] == "training"].reset_index(drop=True)
            args.valid_folds = args.df[args.df['split'] == "test"].reset_index(drop=True)

        if args.inference_mode == 'y':
            cl = 4 if args.label.lower() == "density" else 1
            ### for testing ###
            # model_name = f'{args.model_base_name}_seed_{args.seed}_fold0_best_acc_cancer_ver{args.VER}.pth'
            # model_name = f'{args.model_base_name}_seed_{args.seed}_fold0_best_aucroc_ver{args.VER}.pth'
            # cpath = '/mnt/storage/Devam/mammo-clip-github/checkpoints/RSNA/Classifier/upmc_breast_clip_det_b5_period_n_lp/lr_5e-05_epochs_30_weighted_BCE_y_BIRADS_data_frac_1.0/'
            # model_name = 'upmc_breast_clip_det_b5_period_n_lp_seed_10_fold0_best_val_verBINMLO.pth'
            model_state = torch.load(args.checkpoints, map_location='cpu')
            ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu")
            model = BreastClipClassifier(args, ckpt=ckpt, n_class=cl)
            model.load_state_dict(model_state['model'])
            model = model.to(device)
            args.valid_folds = args.df
            args.mode = 'test'
            args.image_encoder_type = None
            _,test_loader = get_dataloader_RSNA(args)
            y_preds,predictions = test_fn(test_loader, model, args, device)
            _oof_df = args.valid_folds.copy()
            _oof_df['prediction'] = predictions
            print(_oof_df.head(10))
            if len(y_preds)!=0:
                _oof_df['y_preds'] = y_preds
            else:
                _oof_df['y_preds'] = predictions
            # _oof_df['y_preds'] = y_preds    
            # _oof_df = inference_loop(args)
        else:
            _oof_df = train_loop(args, device)

        oof_df = pd.concat([oof_df, _oof_df])
        print(oof_df.head(10))
    if args.dataset.lower() == "rsna" or args.dataset.lower() == "other":
        oof_df = oof_df.reset_index(drop=True)
        if args.label.lower() == "density" or (args.label.lower() == "birads" and not args.binary):
            oof_df['prediction_bin'] = oof_df['y_preds'].apply(lambda x: np.argmax(x, axis=0))
            print(oof_df.head(10))
            print('================ CV ================')
            if not args.binary:
                aucroc = auroc(gt=oof_df[args.label].values, pred=np.array(oof_df['y_preds'].tolist()))
            else:
                aucroc = auroc(gt=oof_df[args.label].values, pred=oof_df['prediction'].values)
            print(f'AUC-ROC: {aucroc}')
            print('\n')
            if args.label.lower() == "density":
                target_names = ['Fatty', 'scattered fibroglandular densities', 'heterogeneously dense', 'extremely dense']
            elif args.label.lower() == "birads":
                if not args.binary:
                    target_names = ['Birads 0', 'Birads 1', 'Birads 2']
                else:
                    target_names = ['Birads 0', 'Birads 1']
            print(classification_report(oof_df[args.label].values, oof_df['prediction_bin'].values, target_names=target_names,digits=4))
        else:
            oof_df['prediction_bin'] = oof_df['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
            if args.label.lower() == "birads":
                target_names = ['Birads 0', 'Birads 1']
                print(classification_report(oof_df[args.label].values, oof_df['prediction_bin'].values, target_names=target_names,digits=4))
            # oof_df_agg = oof_df[['patient_id', 'laterality', args.label, 'prediction', 'fold']].groupby(
            #     ['patient_id', 'laterality']).mean()
            ### for stitched images ###
            oof_df_agg = oof_df


            print(oof_df_agg.head(10))
            print('================ CV ================')
            aucroc = auroc(gt=oof_df_agg[args.label].values, pred=oof_df_agg['prediction'].values)

            oof_df_agg_cancer = oof_df_agg[oof_df_agg[args.label] == 1]
            oof_df_agg_cancer['prediction'] = oof_df_agg_cancer['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
            acc_cancer = compute_accuracy_np_array(oof_df_agg_cancer[args.label].values,
                                                oof_df_agg_cancer['prediction'].values)
            print(f'AUC-ROC: {aucroc}')
            print('\n')
            print(f"Results shape: {oof_df.shape}")
            print('\n')
        print(args.output_path)
        if args.inference_mode == 'y':
            args.output_path = args.output_path / Path('inference/')
            os.makedirs(args.output_path, exist_ok=True)
            oof_df.to_csv(args.output_path / f'seed_{args.seed}_n_folds_{args.n_folds}_test_outputs_aucroc.csv', index=False)
            classification_report_df = pd.DataFrame(classification_report(oof_df[args.label].values, oof_df['prediction_bin'].values, target_names=target_names, output_dict=True)).transpose()
            classification_report_df.to_csv(args.output_path / f'seed_{args.seed}_n_folds_{args.n_folds}_classification_report_aucroc.csv', index=True)
        else:
            oof_df.to_csv(args.output_path / f'seed_{args.seed}_n_folds_{args.n_folds}_outputs.csv', index=False)


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    if args.data_frac < 1.0:
        args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)

    if args.clip_chk_pt_path is not None:
        ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu")
        if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = None
        ckpt = None
    if args.running_interactive:
        # test on small subsets of data on interactive mode
        n_sample = 200
        args.train_folds = args.train_folds.sample(n_sample) #1000
        args.valid_folds = args.valid_folds.sample(n=n_sample) #1000

    train_loader, valid_loader = get_dataloader_RSNA(args)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    model = None
    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "birads" and not args.binary:
        n_class = 3
    else:
        n_class = 1

    optimizer = None
    scheduler = None
    scalar = None
    mapper = None
    attr_embs = None
    if 'breast_clip' in args.arch:
        print(f"Architecture: {args.arch}")
        print(args.image_encoder_type)
        model = BreastClipClassifier(args, ckpt=ckpt, n_class=n_class)
        print("Model is loaded")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.warmup_epochs == 0.1:
            warmup_steps = args.epochs
        elif args.warmup_epochs == 1:
            warmup_steps = len(train_loader)
        else:
            warmup_steps = 10
        lr_config = {
            'total_epochs': args.epochs,
            'warmup_steps': warmup_steps,
            'total_steps': len(train_loader) * args.epochs
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
        scaler = torch.cuda.amp.GradScaler()

    model = model.to(device)
    print(model)

    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')

    if args.label.lower() == "density" or (args.label.lower() == "birads" and not args.binary):
        criterion = torch.nn.CrossEntropyLoss()
    elif args.binary and args.label.lower() == "birads":
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to('cuda')
        print(f'pos_wt: {pos_wt}')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    best_aucroc = 0.
    best_acc = 0
    best_acc_cancer = 0
    best_val = 10000
    for epoch in range(args.epochs):
        start_time = time.time()
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device
        )

        if (
                'efficientnetv2' in args.arch or 'efficientnet_b5_ns' in args.arch
                or 'efficientnet_b5_ns-detect' in args.arch or 'efficientnetv2-detect' in args.arch
        ):
            scheduler.step()

        ## taking extra y_pred for multiclass auc in BIRADS and Density RSNA.
        y_preds, avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, args, device, epoch, mapper=mapper, attr_embs=attr_embs, logger=logger
        )
        args.valid_folds['prediction'] = predictions
        if len(y_preds)!=0:
            args.valid_folds['y_preds'] = y_preds
        else:
            args.valid_folds['y_preds'] = predictions

        valid_agg = None
        if args.dataset.lower() == "vindr":
            valid_agg = args.valid_folds
        elif args.dataset.lower() == "rsna":
            ### for stitched images ###
            valid_agg = args.valid_folds[['patient_id', args.label, 'prediction','y_preds', 'fold']]
            # valid_agg = args.valid_folds[['patient_id', 'laterality', args.label, 'prediction', 'fold']].groupby(
            #     ['patient_id', 'laterality']).mean()

        if args.label.lower() == "density" or (args.label.lower() == "birads" and not args.binary):
            correct_predictions = (valid_agg[args.label] == valid_agg['prediction']).sum()
            total_predictions = len(valid_agg)
            accuracy = correct_predictions / total_predictions
            valid_agg[args.label] = valid_agg[args.label].astype(int)
            valid_agg['prediction'] = valid_agg['prediction'].astype(int)
            f1 = f1_score(valid_agg[args.label], valid_agg['prediction'], average='macro')
            if not args.binary:
                aucroc = auroc(gt=valid_agg[args.label].values, pred=np.array(valid_agg['y_preds'].tolist()))
            else:
                aucroc = auroc(gt=valid_agg[args.label].values, pred=valid_agg['prediction'].values)
            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  '
                f'accuracy: {accuracy * 100:.4f}   f1: {f1 * 100:.4f},'
                f'AUC-ROC Score: {aucroc:.4f} '
            )
            logger.add_scalar(f'valid/{args.label}/accuracy', accuracy, epoch + 1)
            logger.add_scalar(f'valid/{args.label}/AUC-ROC', aucroc, epoch + 1)
            logger.add_scalar(f'valid/{args.label}/f1', f1, epoch + 1)
            logger.add_scalar('valid/epoch-loss', avg_val_loss, epoch + 1)
            
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_val_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save Best Score Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'accuracy': accuracy,
                        'f1': f1,
                        'val_loss': avg_val_loss,
                        'aucroc': aucroc,
                    }, args.chk_pt_path / model_name
                )
            # if best_acc < accuracy:
            #     best_acc = accuracy
            #     model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
            #     print(f'Epoch {epoch + 1} - Save Best acc: {best_acc * 100:.4f} Model')
            #     torch.save(
            #         {
            #             'model': model.state_dict(),
            #             'predictions': predictions,
            #             'epoch': epoch,
            #             'accuracy': accuracy,
            #             'f1': f1,
            #             'aucroc': aucroc,
            #         }, args.chk_pt_path / model_name
            #     )

            #  ## Adding auc roc multiclass (ovr) for validation 

            # if best_aucroc < aucroc:
            #     best_aucroc = aucroc
            #     model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
            #     print(f'Epoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
            #     torch.save(
            #         {
            #             'model': model.state_dict(),
            #             'predictions': predictions,
            #             'epoch': epoch,
            #             'accuracy': accuracy,
            #             'f1': f1,
            #             'aucroc': aucroc,
            #         }, args.chk_pt_path / model_name
            #     )
        else:
            aucroc = auroc(valid_agg[args.label].values, valid_agg['prediction'].values)
            auprc = compute_auprc(valid_agg[args.label].values, valid_agg['prediction'].values)

            valid_agg_cancer = valid_agg[valid_agg[args.label] == 1]
            valid_agg_cancer['prediction'] = valid_agg_cancer['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
            acc_cancer = compute_accuracy_np_array(valid_agg_cancer[args.label].values,
                                                   valid_agg_cancer['prediction'].values)

            elapsed = time.time() - start_time
            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s'
            )
            print(f'Epoch {epoch + 1} - AUC-ROC Score: {aucroc:.4f}')
            logger.add_scalar(f'valid/{args.label}/AUC-ROC', aucroc, epoch + 1)
            logger.add_scalar('valid/epoch-loss', avg_val_loss, epoch + 1)
            logger.add_scalar(f'valid/{args.label}/+ve Acc Score', acc_cancer, epoch + 1)

            
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_val_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save Best Score Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'val_loss': avg_val_loss,
                        'aucroc': aucroc,
                    }, args.chk_pt_path / model_name
                )
            
            # if best_acc_cancer < acc_cancer:
            #     best_acc_cancer = acc_cancer
            #     model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
            #     print(f'Epoch {epoch + 1} - Save Best acc +ve {args.label}: {best_acc_cancer * 100:.4f} Model')
            #     torch.save(
            #         {
            #             'model': model.state_dict(),
            #             'predictions': predictions,
            #             'epoch': epoch,
            #             'auroc': aucroc,
            #         }, args.chk_pt_path / model_name
            #     )

            # if best_aucroc < aucroc:
            #     best_aucroc = aucroc
            #     model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
            #     print(f'Epoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
            #     torch.save(
            #         {
            #             'model': model.state_dict(),
            #             'predictions': predictions,
            #             'epoch': epoch,
            #             'auroc': aucroc,
            #         }, args.chk_pt_path / model_name
            #     )

        if args.label.lower() == "density" or (args.label.lower() == "birads" and not args.binary):
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
            print(f'[Fold{args.cur_fold}], Best Accuracy: {best_acc * 100:.4f}')
        else:
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
            print(
                f'[Fold{args.cur_fold}], AUC-ROC Score: {best_aucroc:.4f}, '
                f'Acc +ve {args.label}: {best_acc_cancer * 100:.4f}'
            )
        predictions = torch.load(args.chk_pt_path / model_name, map_location='cpu')['predictions']
        args.valid_folds['prediction'] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    return args.valid_folds


def inference_loop(args):
    print(f'================== fold: {args.cur_fold} validating ======================')
    print(args.valid_folds.shape)
    predictions = torch.load(
        args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_score_ver084.pth',
        map_location='cpu')['predictions']
    print(f'predictions: {predictions.shape}', type(predictions))
    args.valid_folds['prediction'] = predictions

    valid_agg = args.valid_folds[['patient_id', 'laterality', 'cancer', 'prediction', 'fold']].groupby(
        ['patient_id', 'laterality']).mean()
    aucroc = auroc(valid_agg['cancer'].values, valid_agg['prediction'].values)
    print(f'AUC-ROC: {aucroc}')
    return args.valid_folds.copy()


def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = end = time.time()

    # Initialize loss accumulators and counters
    total_loss = 0.0
    batch_count = 0

    progress_iter = tqdm(enumerate(train_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        if (
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
            inputs = inputs.squeeze(1)

        batch_size = inputs.size(0)
        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({'img': inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                scores = torch.nn.functional.normalize(scores, p=2, dim=1)
                inputs_dict = {'img': inputs, 'scores': scores}
                with torch.cuda.amp.autocast(enabled=args.apex):
                    y_preds = model(inputs_dict)
        else:
            with torch.cuda.amp.autocast(enabled=args.apex):
                y_preds = model(inputs)
        if args.label == "density" or (args.label.lower() == "birads" and not args.binary):
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)
            # Accumulate loss
        total_loss += loss.item() * batch_size
        batch_count += batch_size

        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # batch scheduler
        # scheduler.step()
        if 'breast_clip' in args.arch:
            scheduler.step()
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]['lr']],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}'
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar('train/epoch', epoch, index)
            logger.add_scalar('train/iter_loss', losses.avg, index)
            
            logger.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], index)
    # At the end of the epoch, log the average loss
    avg_epoch_loss = total_loss / (batch_count if batch_count > 0 else 1)
    logger.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
    logger.add_scalar('train/epoch-loss', losses.avg, epoch)
    # Reset accumulators for the next epoch
    total_loss = 0.0
    batch_count = 0


    return losses.avg


def valid_fn(valid_loader, model, criterion, args, device, epoch=1, mapper=None, attr_embs=None, logger=None):
    losses = AverageMeter()
    model.eval()
    preds = []
    y_pred_list=[]
    start = time.time()

    progress_iter = tqdm(enumerate(valid_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                         total=len(valid_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        batch_size = inputs.size(0)
        if (
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
            inputs = inputs.squeeze(1)

        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({'img': inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                inputs_dict = {'img': inputs, 'scores': scores}
                with torch.no_grad():
                    y_preds = model(inputs_dict)
        else:
            with torch.no_grad():
                y_preds = model(inputs)

        if args.label == "density" or (args.label.lower() == "birads" and not args.binary):
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        if args.label == "density" or (args.label.lower() == "birads" and not args.binary):
            _, predicted = torch.max(y_preds, 1)
            preds.extend(predicted.cpu().numpy())
            y_pred_list.extend(y_preds.softmax(dim=1).cpu().numpy())
        else:
            preds.append(y_preds.squeeze(1).sigmoid().to('cpu').numpy())

        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))

        if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and logger is not None:
            index = step + len(valid_loader) * epoch
            logger.add_scalar('valid/iter_loss', losses.avg, index)

    if args.label == "density" or (args.label.lower() == "birads" and not args.binary):
        predictions = np.array(preds)
    else:
        predictions = np.concatenate(preds)

    return y_pred_list,losses.avg, predictions

def test_fn(test_loader, model, args, device ,epoch=0,mapper=None,attr_embs=None):
    model.eval()
    preds = []
    y_pred_list=[]
    start = time.time()
    args.epochs = 1
    progress_iter = tqdm(enumerate(test_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch test]",
                         total=len(test_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        batch_size = inputs.size(0)
        if (
                'efficientnet_b5_ns-detect' in args.arch or
                'efficientnetv2-detect' in args.arch or
                args.arch.lower() == "swin_tiny_tf" or
                args.arch.lower() == 'swin_tiny_custom' or
                args.arch.lower() == "swin_base_tf" or
                args.arch.lower() == 'swin_base_custom' or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp_attn" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_resnet101_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_resnet101_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_resnet101_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_resnet101_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_resnet152_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_resnet152_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_resnet152_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_resnet152_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_tiny_512_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_base_512_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_large_512_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_b5_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_tiny_512_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_base_512_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_large_512_period_n_lp"):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
            inputs = inputs.squeeze(1)

        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({'img': inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                inputs_dict = {'img': inputs, 'scores': scores}
                with torch.no_grad():
                    y_preds = model(inputs_dict)
        else:
            with torch.no_grad():
                y_preds = model(inputs)

        if args.label.lower() == "density" or (args.label.lower() == "birads" and not args.binary):
            _, predicted = torch.max(y_preds, 1)
            preds.extend(predicted.cpu().numpy())
            y_pred_list.extend(y_preds.softmax(dim=1).cpu().numpy())
        else:
            preds.append(y_preds.squeeze(1).sigmoid().to('cpu').numpy())

        progress_iter.set_postfix(
            {
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(test_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  .format(step, len(test_loader),
                          remain=timeSince(start, float(step + 1) / len(test_loader))))
    if args.label.lower() == "density" or (args.label.lower() == "birads" and not args.binary):
        predictions = np.array(preds)
    else:
        predictions = np.concatenate(preds)
    return y_pred_list, predictions

