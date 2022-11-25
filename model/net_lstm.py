import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, time_step, out_point,
                 dropout, hidden_dim, num_layers):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_point = out_point
        self.time_step = time_step
        self.dropout = dropout

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.bn_input = nn.BatchNorm1d(self.time_step)

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True,
                            bidirectional=False)

        # self.bn_mid = nn.BatchNorm1d(self.hidden_dim*2)

        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        # self.classifier = nn.Softmax(dim=1)
        # self.fc = nn.Sequential(nn.Linear(self.hidden_dim*2, self.output_dim),
        #                         nn.ReLU(inplace=True))
        # self.fc2 = nn.Sequential(nn.Linear(self.time_step, self.out_point),
        #                         nn.ReLU(inplace=True))
        # if iscuda:
        #     self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
        #             torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())
        # else:
        #     self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
        #             torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        # self._init_weight()
        # self.apply(self.weight_init)
    # def _init_weight(self):
    #     for m in self.modules():
    #       m.weight = nn.init.xavier_uniform_(m.weight)
            # if isinstance(m, nn.Linear):
            #     nn.init.xavier_normal_(m.weight)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
    # def weight_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)


    def forward(self, input, hidden):
        # input = F.normalize(input, p=2, dim=2)# BxTxC
        input = input.permute(0, 2, 1).contiguous()
        lstm_out, hidden = self.lstm(input, hidden)
        # fcouts = self.fc(lstm_out)#BxTxC
        # fcouts = fcouts.permute(0, 2, 1).contiguous()# BxCxT
        # outs = self.fc2(fcouts)
        outs=[]
        for time_step in range(self.out_point):    # calculate output for each time step
            outs.append(self.fc(F.normalize(lstm_out[:, -time_step, :], p=2, dim=1)))

        output = torch.stack(outs, dim=1).permute(0,2,1).contiguous()


        return output[:,:,0],hidden


if __name__ == "__main__":
    model = LSTM(input_dim=16, output_dim=6, time_step=1, out_point=1,
                 dropout=0.4,
                 hidden_dim=128,
                 num_layers=5)
    print(model)
    input = torch.randn([64, 16, 300])
    hidden = None
    output,hidden = model(input,hidden)
    print(hidden)
    print(output)
    output,hidden = model(input,hidden)
    print(hidden)
    print(output)
    print(output.shape)