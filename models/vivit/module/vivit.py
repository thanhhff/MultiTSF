import torch
import torch.nn as nn
import math
from einops import rearrange, reduce, repeat
from models.vivit.module.transformer import PatchEmbed, TransformerContainer, get_sine_cosine_pos_emb
from models.vivit.module.weight_init import (trunc_normal_, init_from_vit_pretrain_, 
	init_from_mae_pretrain_, init_from_kinetics_pretrain_)


class ViViT(nn.Module):
	"""ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
		<https://arxiv.org/abs/2103.15691>

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to 12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0..
		tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
			Defaults to 'fact_encoder'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'fact_encoder', 'joint_space_time', 'divided_space_time'
	]

	def __init__(self,
				 num_frames,
				 img_size=224,
				 patch_size=16,
				 pretrain_pth=None,
				 weights_from='imagenet',
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=12,
				 in_channels=3,
				 dropout_p=0.,
				 tube_size=2,
				 conv_type='Conv3d',
				 attention_type='fact_encoder',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 extend_strategy='temporal_avg',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')
		
		num_frames = num_frames//tube_size
		self.num_frames = num_frames
		self.pretrain_pth = pretrain_pth
		self.weights_from = weights_from
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.conv_type = conv_type
		self.copy_strategy = copy_strategy
		self.extend_strategy = extend_strategy
		self.tube_size = tube_size
		self.num_time_transformer_layers = 0
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token

		#tokenize & position embedding
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=embed_dims,
			tube_size=tube_size,
			conv_type=conv_type)
		num_patches = self.patch_embed.num_patches
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention - Model 3
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container
		elif self.attention_type == 'joint_space_time':
			# Joint Space Time Attention - Model 1
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)
			
			transformer_layers = container
		else:
			# Divided Space Time Transformer Encoder - Model 2
			transformer_layers = nn.ModuleList([])
			self.num_time_transformer_layers = 4
			
			spatial_transformer = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])
			
			temporal_transformer = TransformerContainer(
				num_transformer_layers=self.num_time_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])

			transformer_layers.append(spatial_transformer)
			transformer_layers.append(temporal_transformer)
 
		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_enb
		if attention_type == 'fact_encoder':
			num_frames = num_frames + 1
			num_patches = num_patches + 1
			self.use_cls_token_temporal = False
		else:
			self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
			if self.use_cls_token_temporal:
				num_frames = num_frames + 1
			else:
				num_patches = num_patches + 1

		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
			self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
			self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		self.drop_after_time = nn.Dropout(p=dropout_p)

		self.init_weights()

	def init_weights(self):
		if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			#trunc_normal_(self.time_embed, std=.02)
			nn.init.trunc_normal_(self.pos_embed, std=.02)
			nn.init.trunc_normal_(self.time_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		
		if self.pretrain_pth is not None:
			if self.weights_from == 'imagenet':
				init_from_vit_pretrain_(self,
										self.pretrain_pth,
										self.conv_type,
										self.attention_type,
										self.copy_strategy,
										self.extend_strategy, 
										self.tube_size, 
										self.num_time_transformer_layers)
			elif self.weights_from == 'kinetics':
				init_from_kinetics_pretrain_(self,
											 self.pretrain_pth)
			else:
				raise TypeError(f'not support the pretrained weight {self.pretrain_pth}')

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def prepare_tokens(self, x):
		#Tokenize
		b = x.shape[0]
		x = self.patch_embed(x)
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
		
		x = self.drop_after_pos(x)

		# Add Time Embedding
		if self.attention_type != 'fact_encoder':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens,
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
		
		return x, cls_tokens, b

	def forward(self, x):
		x, cls_tokens, b = self.prepare_tokens(x)

		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')

			import pdb; pdb.set_trace()
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			
			x = temporal_transformer(x)

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)

	def get_last_selfattention(self, x):
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x, return_attention=True)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)

			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			print(x.shape)
			x = temporal_transformer(x, return_attention=True)
		return x


if __name__ == '__main__':
    num_frames = 8
    num_frames = num_frames * 2
    model = ViViT(num_frames=num_frames,
                  img_size=224,
                  patch_size=16,
                  embed_dims=768,
                  in_channels=3,
                  attention_type='fact_encoder',
                  return_cls_token=True)


    import pdb; pdb.set_trace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    x = torch.randn(2, num_frames, 3, 224, 224).to(device)
    model.to(device)
    out = model(x)
    print(out.shape)
