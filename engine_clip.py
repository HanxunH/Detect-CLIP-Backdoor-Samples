import torch
import lid
import util
import misc
import time
import math
import copy
import torch.nn.functional as F
from lid import gmean
from datasets.zero_shot_metadata import zero_shot_meta_dict
from tqdm import tqdm

@torch.no_grad()
def track_training_loss(model, criterion, scaler, train_loader, data, hf, args):
    device = args.gpu
    # Track training Loss for ABL in backdoor mode
    model.eval()
    criterion.reduction = False
    data.train_set.get_idx = True
    for idxs, images, texts in tqdm(train_loader):
        images, texts = images.to(device, non_blocking=True), texts.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = criterion(model, (images, texts))
        loss = results['loss']
        idxs = idxs.tolist()
        
        batch_results = {}
        for i, idx in enumerate(idxs):
            batch_results[idx] = loss[i].item()
        
        if misc.world_size() > 1:
            full_rank_results = misc.all_gather(batch_results)
        else:
            full_rank_results = [batch_results]

        if misc.get_rank() == 0:
            for rank_result in full_rank_results:
                for idx, loss in rank_result.items():
                    hf['data'][idx] = loss

    criterion.reduction = True
    data.train_set.get_idx = False
    return hf

@torch.no_grad()
def evaluate_backdoor_asr(model, loader, args, configs, tokenizer):
    # Evaluate:
    model.eval()
    device = args.gpu

    classnames = zero_shot_meta_dict[configs.class_names]
    templates = zero_shot_meta_dict[configs.zero_shot_templates]
    use_format = isinstance(templates[0], str)
    zeroshot_weights = []
    for classname in classnames:
        texts = [template.format(classname) if use_format else template(classname) for template in templates]
        texts = tokenizer(texts).to(device)  # tokenize
        if args.ddp:
            class_embeddings = model.module.encode_text(texts)
        else:
            class_embeddings = model.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    # extract features
    metric_logger = misc.MetricLogger(delimiter="  ")
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(images)
        image_features = out['image_features']
        logits = 100. * image_features @ zeroshot_weights
        acc1, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        metric_logger.update(acc1=acc1.item(), n=images.shape[0])
        metric_logger.update(acc5=acc5.item(), n=images.shape[0])
    
    metric_logger.synchronize_between_processes()
    results = {
        "bd_test_acc1": metric_logger.meters['acc1'].global_avg,
        "bd_test_acc5": metric_logger.meters['acc5'].global_avg,
    }
    return results

@torch.no_grad()
def evaluate(model, loader, args, configs, tokenizer):
    # Evaluate:
    model.eval()
    device = args.gpu

    templates = zero_shot_meta_dict[configs.zero_shot_templates]
    classnames = zero_shot_meta_dict[configs.class_names]
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    zeroshot_weights = []
    for classname in classnames:
        texts = [template.format(classname) if use_format else template(classname) for template in templates]
        texts = tokenizer(texts).to(device)  # tokenize
        if args.ddp:
            class_embeddings = model.module.encode_text(texts)
        else:
            class_embeddings = model.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    
    # extract features
    metric_logger = misc.MetricLogger(delimiter="  ")
    for data in loader:
        if len(data) == 2:
            images, labels = data
        elif len(data) == 3:
            idxs, images, labels = data
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(images)
        image_features = out['image_features']
        logits = 100. * image_features @ zeroshot_weights
        acc1, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        metric_logger.update(acc1=acc1.item(), n=images.shape[0])
        metric_logger.update(acc5=acc5.item(), n=images.shape[0])
        if args.ddp:
            full_image_features = torch.cat(misc.gather(image_features), dim=0)
        else:
            full_image_features = image_features
        
        image_lid_k32 = lid.lid_mle(data=image_features.detach(), reference=full_image_features.detach(), k=32)
        image_lid_k512 = lid.lid_mle(data=image_features.detach(), reference=full_image_features.detach(), k=512)
        metric_logger.update(image_lid_k32_avg=image_lid_k32.mean().item(), n=images.shape[0])
        metric_logger.update(image_lid_k512_avg=image_lid_k512.mean().item(), n=images.shape[0])
        metric_logger.update(image_lid_k32_var=image_lid_k32.var().item(), n=images.shape[0])
        metric_logger.update(image_lid_k512_var=image_lid_k512.var().item(), n=images.shape[0])
        metric_logger.update(image_lid_k32_gavg=gmean(image_lid_k32).item(), n=images.shape[0])
        metric_logger.update(image_lid_k512_gavg=gmean(image_lid_k512).item(), n=images.shape[0])

    metric_logger.synchronize_between_processes()
    results = {
        "test_acc1": metric_logger.meters['acc1'].global_avg,
        "test_acc5": metric_logger.meters['acc5'].global_avg,
        "test_image_lid_k32_avg": metric_logger.meters['image_lid_k32_avg'].global_avg,
        "test_image_lid_k512_avg": metric_logger.meters['image_lid_k512_avg'].global_avg,
        "test_image_lid_k32_var": metric_logger.meters['image_lid_k32_var'].global_avg,
        "test_image_lid_k512_var": metric_logger.meters['image_lid_k512_var'].global_avg,
        "test_image_lid_k32_gavg": metric_logger.meters['image_lid_k32_gavg'].global_avg,
        "test_image_lid_k512_gavg": metric_logger.meters['image_lid_k512_gavg'].global_avg,
    }
    return results

def train_epoch(exp, model, optimizer, criterion, scaler, train_loader, global_step, epoch, logger, args, use_global_step=False, step_per_epoch=0):
    epoch_stats = {}
    device = args.gpu
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # Training
    model.train()
    for i, data in enumerate(train_loader):
        start = time.time()
        # Adjust LR
        if use_global_step:
            util.adjust_learning_rate(optimizer, global_step / step_per_epoch, exp.config)
        else:
            util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        # Train step
        model.train()
        optimizer.zero_grad()
        if len(data) == 2:
            data = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
        elif len(data) == 3:
            data = data[0], data[1].to(device, non_blocking=True), data[2].to(device, non_blocking=True)
        elif len(data) == 4:
            data = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True), data[2].to(device, non_blocking=True), data[3].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = criterion(model, data)
            loss = results['loss']
            logits = results['logits']
            labels = results['labels']
        # Optimize
        loss = results['loss']
        if torch.isnan(loss):
            if misc.get_rank() == 0:
                logger.info('Skip this batch, loss is nan!')
            raise('loss is nan!')
        if scaler is not None:
            scaler.scale(loss).backward()
            if hasattr(exp.config, 'grad_clip'):
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                if hasattr(exp.config, 'grad_clip'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp.config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if hasattr(exp.config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), exp.config.grad_clip)
            optimizer.step()
        torch.cuda.synchronize()
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            util.unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        loss = loss.item()
        # Calculate acc
        acc, _ = util.accuracy(logits, labels, topk=(1, 5))
        # Update Meters
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc.item(), n=batch_size)
        metric_logger.update(logits_scale=results['logits_scale'].mean().item())
        metric_logger.update(vision_lids_k32_avg=results['vision_lids_k32'].mean().item())
        metric_logger.update(vision_lids_k32_var=results['vision_lids_k32'].var().item())
        metric_logger.update(vision_lids_k512_avg=results['vision_lids_k512'].mean().item())
        metric_logger.update(vision_lids_k512_var=results['vision_lids_k512'].var().item())
        metric_logger.update(vision_lids_k32_gavg=gmean(results['vision_lids_k32']).item())
        metric_logger.update(vision_lids_k512_gavg=gmean(results['vision_lids_k512']).item())
        metric_logger.update(text_lids_k32_avg=results['text_lids_k32'].mean().item())
        metric_logger.update(text_lids_k32_var=results['text_lids_k32'].var().item())
        metric_logger.update(text_lids_k512_avg=results['text_lids_k512'].mean().item())
        metric_logger.update(text_lids_k512_var=results['text_lids_k512'].var().item())
        metric_logger.update(text_lids_k32_gavg=gmean(results['text_lids_k32']).item())
        metric_logger.update(text_lids_k512_gavg=gmean(results['text_lids_k512']).item())
        metric_logger.update(main_loss=results['main_loss'])
        if 'adaptive_loss' in results:    
            metric_logger.update(adaptive_loss=results['adaptive_loss'])
        if 'reg_loss' in results:
            metric_logger.update(reg_loss=results['reg_loss'])
        if 'I_M' in results:
            metric_logger.update(I_M=results['I_M'])
        # Log results
        end = time.time()
        time_used = end - start
        # track LR
        lr = optimizer.param_groups[0]['lr']
        if global_step % exp.config.log_frequency == 0:
            metric_logger.synchronize_between_processes()
            payload = {
                "lr": lr,
                'logits_scale': metric_logger.meters['logits_scale'].avg,
                "acc_avg": metric_logger.meters['acc'].avg,
                "v_lid32_gavg": metric_logger.meters['vision_lids_k32_gavg'].avg,
                "v_lid512_gavg": metric_logger.meters['vision_lids_k512_gavg'].avg,
                "t_lid32_gavg": metric_logger.meters['text_lids_k32_gavg'].avg,
                "t_lid512_gavg": metric_logger.meters['text_lids_k512_gavg'].avg,
                "loss_avg": metric_logger.meters['loss'].avg,
                "main_loss": metric_logger.meters['main_loss'].avg,
            }
            if 'adaptive_loss' in results:
                payload['adaptive_loss'] = metric_logger.meters['adaptive_loss'].avg
            if 'reg_loss' in results:
                payload['reg_loss'] = metric_logger.meters['reg_loss'].avg
            if 'I_M' in results:
                payload['I_M'] = metric_logger.meters['I_M'].avg
            if misc.get_rank() == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=global_step,
                                           time_elapse=time_used,
                                           **payload)
                logger.info(display)
        # Update Global Step
        global_step += 1
        
    metric_logger.synchronize_between_processes()
    epoch_stats['epoch'] = epoch
    epoch_stats['global_step'] = global_step
    epoch_stats['train_acc'] = metric_logger.meters['acc'].global_avg
    epoch_stats['train_loss'] = metric_logger.meters['loss'].global_avg
    epoch_stats['logits_scale'] = metric_logger.meters['logits_scale'].global_avg
    epoch_stats['train_vision_lid32_avg'] = metric_logger.meters['vision_lids_k32_avg'].global_avg
    epoch_stats['train_vision_lid32_var'] = metric_logger.meters['vision_lids_k32_var'].global_avg
    epoch_stats['train_vision_lid512_avg'] = metric_logger.meters['vision_lids_k512_avg'].global_avg
    epoch_stats['train_vision_lid512_var'] = metric_logger.meters['vision_lids_k512_var'].global_avg
    epoch_stats['train_vision_lid32_gavg'] = metric_logger.meters['vision_lids_k32_gavg'].global_avg
    epoch_stats['train_vision_lid512_gavg'] = metric_logger.meters['vision_lids_k512_gavg'].global_avg
    epoch_stats['train_text_lid32_avg'] = metric_logger.meters['text_lids_k32_avg'].global_avg
    epoch_stats['train_text_lid32_var'] = metric_logger.meters['text_lids_k32_var'].global_avg
    epoch_stats['train_text_lid512_avg'] = metric_logger.meters['text_lids_k512_avg'].global_avg
    epoch_stats['train_text_lid512_var'] = metric_logger.meters['text_lids_k512_var'].global_avg
    epoch_stats['train_text_lid32_gavg'] = metric_logger.meters['text_lids_k32_gavg'].global_avg
    epoch_stats['train_text_lid512_gavg'] = metric_logger.meters['text_lids_k512_gavg'].global_avg
    epoch_stats['main_loss'] = metric_logger.meters['main_loss'].global_avg
    if 'adaptive_loss' in results:
        epoch_stats['adaptive_loss'] = metric_logger.meters['adaptive_loss'].global_avg
    if 'reg_loss' in results:
        epoch_stats['reg_loss'] = metric_logger.meters['reg_loss'].global_avg
    if 'I_M' in results:   
        epoch_stats['I_M'] = metric_logger.meters['I_M'].global_avg

    return epoch_stats


