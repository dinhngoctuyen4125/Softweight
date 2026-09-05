import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.covariance
from tqdm import tqdm
from transformers import RobertaForMaskedLM


class CodeBERTForSelector(nn.Module):
    """Frozen CodeBERT used only to extract features from its 13 hidden layers.

    The checkpoint (tummitum/codebert-deprecated) is already fine-tuned, so no
    training happens here: we only read out h(x) = {h1(x), ..., h13(x)}.
    """

    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.layer_num = 13
        self.device = device
        self.roberta = RobertaForMaskedLM.from_pretrained(model_name, output_hidden_states=True)

    @torch.no_grad()
    def sample_X_estimator(self, dataloader):
        """Per layer: centroid, precision (inverse covariance) and normalised feature bank of D_forget."""
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

        all_layer_features = []
        num_layers = self.layer_num
        for i in range(num_layers):
            all_layer_features.append([])

        for step, batch in enumerate(tqdm(dataloader, desc="extract features")):
            self.eval()
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                all_layer_features[i].append(layer_mean_fea.data.cpu())

        mean_list = []
        precision_list = []
        fea_list = []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            fea_list.append(F.normalize(all_layer_features[i], dim=-1))
            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()
            mean_list.append(sample_mean.to(self.device))
            precision_list.append(temp_precision.to(self.device))

        return mean_list, precision_list, fea_list

    @torch.no_grad()
    def get_unsup_Mah_score(self, dataloader, sample_mean, precision, fea_list):
        """Score vector s(x): per layer, Mahalanobis distance to the D_forget centroid
        combined with minus the largest cosine similarity to any record of D_forget.
        Returns an array of shape [num_records, 13]."""
        total_mah_scores = []
        num_layers = self.layer_num
        for i in range(num_layers):
            total_mah_scores.append([])

        # move the feature bank to the device once instead of on every batch
        fea_list = [fea.t().to(self.device) for fea in fea_list]

        for step, batch in enumerate(tqdm(dataloader, desc="score")):
            batch_all_features = []
            self.eval()
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                batch_all_features.append(layer_mean_fea.data)

            for i in range(len(batch_all_features)):
                batch_sample_mean = sample_mean[i]
                out_features = batch_all_features[i]
                zero_f = out_features - batch_sample_mean
                gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
                out_feas = F.normalize(out_features, dim=-1)
                cs_score = out_feas @ fea_list[i]
                cs_score = torch.max(cs_score, dim=1)[0]
                all_score = -cs_score * 1000. + gaussian_score
                total_mah_scores[i].extend(all_score.cpu().numpy())

        for i in range(len(total_mah_scores)):
            total_mah_scores[i] = np.expand_dims(np.array(total_mah_scores[i]), axis=1)

        return np.concatenate(total_mah_scores, axis=1)
