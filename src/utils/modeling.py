import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification, AutoModelForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
from transformers import RobertaModel, RobertaConfig, AlbertModel, AutoModel, AutoConfig


class roberta_large_classifier(nn.Module):

    def __init__(self, num_labels, model_select, dropout):

        super(roberta_large_classifier, self).__init__()

        self.config = AutoConfig.from_pretrained('../../model_hub/bertweet-large', local_files_only=True)
        self.roberta = AutoModel.from_pretrained('../../model_hub/bertweet-large', local_files_only=True)
        # self.config = AutoConfig.from_pretrained('vinai/bertweet-large')
        # self.roberta = AutoModel.from_pretrained('vinai/bertweet-large')
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.config.hidden_size*2, self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        
        # Avg text and target tokens
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        
        eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        assert len(eos_token_ind) == 3*len(kwargs['input_ids'])
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        x_atten_clone = x_atten_masks.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out