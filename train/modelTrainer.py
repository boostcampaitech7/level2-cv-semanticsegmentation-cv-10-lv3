import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import datetime
from utils import log_to_file, save_model, dice_coef
import configs.config as cfg
from tqdm.auto import tqdm
from dataset.XRayDataset import CLASSES


def train_model(model, data_loader, val_loader, criterion, optimizer, scheduler):
    scaler = GradScaler()
    best_loss = float('inf') if all else 0.
    model = model.cuda()

    num_epochs = cfg.NUM_EPOCHS
    log_path = cfg.LOG_NAME

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0 # all인 경우에만 사용

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if all:
                total_loss += loss.item()

            if (step + 1) % 25 == 0:
                log_message = (
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(), 4)}'
                )
                print(log_message)
                log_to_file(log_message, log_path)

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        if epoch % 10 == 0:
            log_message = f"Epoch {epoch + 1}, Current LR: {current_lr:.6f}"
            print(log_message)
            log_to_file(log_message)
            
        if all:
            avg_loss = total_loss / len(data_loader)
            log_message = f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}"
            print(log_message)
            log_to_file(log_message, log_path)

            if avg_loss < best_loss:
                log_message = (
                    f"Best performance at epoch: {epoch + 1}, Loss {best_loss:.4f} -> {avg_loss:.4f}\n"
                    f"Save model in {cfg.SAVED_DIR}"
                )
                print(log_message)
                log_to_file(log_message, log_path)
                best_loss = avg_loss
                save_model(model)
        else:
            # Validation
            if (epoch + 1) % VAL_EVERY == 0:
                dice = validation(epoch + 1, model, val_loader, criterion)

                if best_loss < dice:
                    log_message = (
                        f"Best performance at epoch: {epoch + 1}, {best_loss:.4f} -> {dice:.4f}\n"
                        f"Save model in {SAVED_DIR}"
                    )
                    print(log_message)
                    log_to_file(log_message)
                    best_loss = dice
                    save_model(model)

def validation(epoch, model, data_loader, criterion, thr=cfg.TH):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            if cfg.MODEL.lower() == "fcn":
                outputs = model(images)['out']
            else:
                outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    log_to_file(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    return avg_dice