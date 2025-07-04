import torch
import torch.nn as nn

class MarkovBlanketAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MarkovBlanketAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_s_layer = nn.Linear(input_dim, hidden_dim)
        self.key_ss_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_sb_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_zb_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_s_layer = nn.Linear(hidden_dim, hidden_dim)
        
        self.query_b_layer = nn.Linear(input_dim, hidden_dim)
        self.key_bb_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_zb_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_b_layer = nn.Linear(hidden_dim, hidden_dim)
        
        self.query_z_layer = nn.Linear(input_dim, hidden_dim)
        self.key_zz_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_z_layer = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, y, s, b, z, s_atten_bias, b_atten_bias, z_atten_bias):
        batch_size = y.size(0)
        
        # Compute query tensors
        query_s = self.query_s_layer(y)
        query_b = self.query_b_layer(y)
        query_z = self.query_z_layer(y)
        
        # Compute key and value tensors for s
        key_s = self.key_ss_layer(s) + self.key_sb_layer(b)
        value_s = self.value_s_layer(s)
        
        # Compute key and value tensors for b
        key_b = self.key_bb_layer(b) + self.key_sb_layer(s) + self.key_zb_layer(z)
        value_b = self.value_b_layer(b)
        
        # Compute key and value tensors for z
        key_z = self.key_zz_layer(z) + self.key_bz_layer(b)
        value_z = self.value_z_layer(z)
        
        # Split tensors for multi-head attention
        query_s = query_s.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_s = key_s.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_s = value_s.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        query_b = query_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_b = key_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_b = value_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        query_z = query_z.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_z = key_z.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_z = value_z.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores_s = torch.matmul(query_s, key_s.transpose(-2, -1))
        attention_scores_b = torch.matmul(query_b, key_b.transpose(-2, -1))
        attention_scores_z = torch.matmul(query_z, key_z.transpose(-2, -1))
        
        # Apply attention biases and compute attention weights
        attention_weights_s = self.softmax(attention_scores_s + s_atten_bias)
        attention_weights_b = self.softmax(attention_scores_b + b_atten_bias)
        attention_weights_z = self.softmax(attention_scores_z + z_atten_bias)
        
        # Compute weighted sum of values
        weighted_sum_s = torch.matmul(attention_weights_s, value_s)
        weighted_sum_b = torch.matmul(attention_weights_b, value_b)
        weighted_sum_z = torch.matmul(attention_weights_z, value_z)
        
        # Reshape and apply linear transformation to obtain output tensors
        s_output = s + self.out_layer(weighted_sum_s.transpose(1, 2).contiguous().view(batch_size, -1, hidden_dim))
        b_output = s_output + b + self.out_layer(weighted_sum_b.transpose(1, 2).contiguous().view(batch_size, -1, hidden_dim))
        z_output = b_output + z + self.out_layer(weighted_sum_z.transpose(1, 2).contiguous().view(batch_size, -1, hidden_dim))
        
        return s_output, b_output, z_output, attention_scores_s, attention_scores_b, attention_scores_z


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            MarkovBlanketAttention(input_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        
    def forward(self, y, mask):
        s_atten_bias = torch.zeros(batch_size, seq_length, seq_length)
        b_atten_bias = torch.zeros(batch_size, seq_length, seq_length)
        z_atten_bias = torch.zeros(batch_size, seq_length, seq_length)
        
        s = torch.zeros(batch_size, seq_length, hidden_dim)
        b = torch.zeros(batch_size, seq_length, hidden_dim)
        z = torch.zeros(batch_size, seq_length, hidden_dim)        

        for layer in self.layers:
            s, b, z, s_atten_bias, b_atten_bias, z_atten_bias = layer(y, s, b, z, s_atten_bias, b_atten_bias, z_atten_bias)
        
        return s, b, z

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        
        self.linear_b = nn.Linear(hidden_dim, 3)
        self.softmax = nn.Softmax(dim=-1)
        
        self.linear_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.linear_b_dec = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.linear_b_transformed = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.linear_z = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, s, b, z):
        # Linear transformations for s, b, and z
        s_linear = s
        b_dec_linear = b
        z_linear = z
        
        # Apply rectified linear activation (ReLU) to s, b, and z for each layer
        for layer in range(self.num_layers):
            s_linear = F.relu(self.linear_s[layer](s_linear))
            b_dec_linear = F.relu(self.linear_b_dec[layer](b_dec_linear))
            z_linear = F.relu(self.linear_z[layer](z_linear))
        
        # Apply linear transformation to b_transformed
        b_transformed = b
        for layer in range(self.num_layers):
            b_transformed = F.relu(self.linear_b_transformed[layer](b_transformed))
        
        b_transformed = self.linear_b(b_transformed)
        
        # Combine the ReLU transformations with the softmax output
        combined = s_linear + b_dec_linear + z_linear + b_transformed
        
        # Apply linear transformation to obtain the output
        output = self.linear_out(combined)
        
        return output


class DeepDMBD(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, num_heads, num_layers)
        self.decoder = Decoder(hidden_dim, output_dim, num_layers)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def train_model(self, input_data, target_data, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            encoded_s, encoded_b, encoded_z = self.encoder(input_data)
            output = self.decoder(encoded_s, encoded_b, encoded_z)
            
            # Compute loss
            loss = self.criterion(output, target_data)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
              # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Example usage (commented out to avoid module-level execution):
# input_dim = 128
# output_dim = 64
# hidden_dim = 256
# num_heads = 8
# num_layers = 6
# model = DeepDMBD(input_dim, output_dim, hidden_dim, num_heads, num_layers)
# model.train_model(input_data, target_data, num_epochs=100)
