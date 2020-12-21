import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
import torch.nn.functional as F

class Epoch:

    def __init__(self, model, arch, loss, metrics, stage_name, shot, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.shot = shot
        self.arch = arch

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for query_img, query_mask, support_img, support_mask, history_mask, sample_class, index in iterator:
                
                query_img, query_mask, support_img, support_mask, history_mask, sample_class, index = \
                query_img.to(self.device), query_mask.to(self.device), support_img.to(self.device), \
                support_mask.to(self.device), history_mask.to(self.device), sample_class.to(self.device), index.to(self.device)

                loss, y_pred = self.batch_update(dataloader, query_img, query_mask, support_img, \
                                                support_mask, history_mask, sample_class, index)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, query_mask).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs, query_img, query_mask, y_pred


class TrainEpoch(Epoch):

    def __init__(self, model, arch, loss, metrics, optimizer, shot, device='cpu', verbose=True):
        super().__init__(
            model=model,
            arch=arch,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            shot=shot,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, dataloader, query_img, query_mask, support_img, support_mask, history_mask, sample_class, index):
        self.optimizer.zero_grad()
        b, c, w, h = query_mask.size()
        print('shot: {0}, arch: {1}'.format(self.shot, self.arch))
        prediction = self.model(query_rgb, support_rgb, support_mask, history_mask)
        prediction = nn.functional.interpolate(prediction, size=(w, h), mode='bilinear', align_corners=True)
        
        for j in range (support_mask.shape[0]):
            sub_index=index[j]
            dataloader.history_mask_list[sub_index]=pred_softmax[j]

        loss = self.loss(prediction, query_mask)
        
        loss.backward()
        self.optimizer.step()
        
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, arch, loss, metrics, shot, device='cpu', verbose=True):
        super().__init__(
            model=model,
            arch=arch,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            shot=shot,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, dataloader, query_img, query_mask, support_img, support_mask, history_mask, sample_class, index):
        
        with torch.no_grad():
            print('shot: {0}, arch: {1}'.format(self.shot, self.arch))
            valset.history_mask_list=[None] * 1000
            b, c, w, h = query_mask.size()
            prediction = self.model(query_rgb, support_rgb, support_mask, history_mask)
            prediction = nn.functional.interpolate(prediction, size=(w, h), mode='bilinear', align_corners=True)
            
            for j in range (support_mask.shape[0]):
                sub_index=index[j]
                dataloader.history_mask_list[sub_index]=pred_softmax[j]
            prediction = F.upsample(prediction, size=(w, h), mode='bilinear')
            loss = self.loss(prediction, query_mask) + energy_fg + energy_bg
            # query_img = F.upsample(query_img, size=(size[0], size[1]), mode='bilinear')
            # query_mask = F.upsample(query_mask, size=(size[0], size[1]), mode='nearest')

        return loss, prediction