import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout = 0,  batch_first=True)
        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)    
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        captions = captions[:, :-1] # get rid of <end>
        embeds = self.word_embeddings(captions)    
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        lstm_out, _ = self.lstm(embeds)
        tag_outputs  = self.hidden2tag(lstm_out)
        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        words = []   
        words_size = 0
        
        while (words_size != max_len+1):
            # inputs (1,1,embed_size)
            # output (1,1,hidden_size)
            output, states = self.lstm(inputs,states)   
            # output.squeeze(dim = 1) (1,hidden_size)
            # output (1,vocab_size)
            output = self.hidden2tag(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            # back to host
            words.append(predicted_index.cpu().numpy()[0].item())   
            # Look for <end> in the vocabulary 
            if (predicted_index == 1):
                break    
            # Go to next time step
            inputs = self.word_embeddings(predicted_index)   
            inputs = inputs.unsqueeze(1) 
            words_size += 1
        return words