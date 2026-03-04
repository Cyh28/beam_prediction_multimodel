# utils.py
import torch


def topk_accuracy(output, target, k=1):
    with torch.no_grad():
        _, pred = output.topk(k, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.sum().item() / target.size(0)


def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0

    with torch.no_grad():
        for image, geo, y in dataloader:
            image = image.to(device)
            geo = geo.to(device)
            y = y.to(device)

            output = model(image, geo)

            batch_size = y.size(0)
            total += batch_size

            top1 += topk_accuracy(output, y, k=1) * batch_size
            top3 += topk_accuracy(output, y, k=3) * batch_size
            top5 += topk_accuracy(output, y, k=5) * batch_size

    return top1/total, top3/total, top5/total