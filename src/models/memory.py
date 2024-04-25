import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        
        m, d = mem.size()
        if train:
            query_update = torch.zeros((m,d)).cuda()
            # query_update = torch.zeros((m,d))
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0

            return query_update 
    
        else:
            query_update = torch.zeros((m,d)).cuda()
            # query_update = torch.zeros((m,d))
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0 
            
            return query_update

    def get_score(self, mem, query):
        bs, h,w,d = query.size()
        m, d = mem.size()
        
        score = torch.matmul(query, torch.t(mem))# b X h X w X m
        score = score.view(bs*h*w, m)# (b X h X w) X m
        
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)
        
        return score_query, score_memory
    
    def forward(self, query, keys, train=True):

        batch_size, dims,h,w = query.size() # b X d X h X w
        query = F.normalize(query, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        
        #train
        if train:
            #gathering loss
            gathering_loss = self.gather_loss(query,keys, train)
            #spreading_loss
            spreading_loss = self.spread_loss(query, keys, train)
            # read
            updated_query, softmax_score_query,softmax_score_memory = self.read(query, keys)
            #update
            updated_memory = self.update(query, keys, train)
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
        
        #test
        else:
            #gathering loss
            gathering_loss = self.gather_loss(query,keys, train)
            #spreading_loss
            spreading_loss = self.spread_loss(query, keys, train)
            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            
            #update
            updated_memory = keys
                
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss

        
    def update(self, query, keys, train):
        
        batch_size, h,w,dims = query.size() # b X h X w X d 
        
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        
        if train:
            # top-1 queries (of each memory) update (weighted sum) & random pick 
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        
        else:
            # only weighted sum update when test 
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        
        # top-1 update
        #query_update = query_reshape[updating_indices][0]
        #updated_memory = F.normalize(query_update + keys, dim=1)
      
        return updated_memory.detach()
        

    def spread_loss(self,query, keys, train):
        batch_size, h,w,dims = query.size() # b X h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        #1st, 2nd closest memories
        pos = keys[gathering_indices[:,0]]
        neg = keys[gathering_indices[:,1]]

        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())

        return spreading_loss
        
    def gather_loss(self, query, keys, train):
        
        batch_size, h,w,dims = query.size() # b X h X w X d

        loss_mse = torch.nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss
        
        
    def read(self, query, updated_memory):
        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory) # (b X h X w) X d
        updated_query = torch.cat((query_reshape, concat_memory), dim = 1) # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0,3,1,2)
        
        return updated_query, softmax_score_query, softmax_score_memory
    