import torch


def compute_topk(output, target, topk=(1, 3, 5)):
    """
    更高效的 top-k accuracy 计算
    只进行一次 topk 操作
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # shape: (maxk, B)

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item() / batch_size)

    return res


def evaluate(model, dataloader, device, criterion=None):
    model.eval()

    total_samples = 0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_loss = 0.0

    with torch.no_grad():
        for image, geo, radar, y in dataloader:
            image = image.to(device)
            geo = geo.to(device)
            radar = radar.to(device)
            y = y.to(device)

            output = model(image, geo, radar)

            batch_size = y.size(0)
            total_samples += batch_size

            if criterion is not None:
                loss = criterion(output, y)
                total_loss += loss.item() * batch_size

            top1, top3, top5 = compute_topk(output, y)

            total_top1 += top1 * batch_size
            total_top3 += top3 * batch_size
            total_top5 += top5 * batch_size

    avg_top1 = total_top1 / total_samples
    avg_top3 = total_top3 / total_samples
    avg_top5 = total_top5 / total_samples

    if criterion is None:
        return avg_top1, avg_top3, avg_top5

    avg_loss = total_loss / total_samples
    return avg_loss, avg_top1, avg_top3, avg_top5