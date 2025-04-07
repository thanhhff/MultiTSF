
class ModelFactory:
    @staticmethod
    def get_model(model_name, config):
        if model_name == "CNN2D":
            from models.cnn2d.cnn2d import CNN2D
            return CNN2D(config.video_encoder, config.len_feature, config.num_classes, config.num_segments, config.fusion_type)
        
        elif model_name == "CNN2D_Transformer":
            from models.cnn2d_transformer.cnn2d_transformer import CNN2D_Transformer
            return CNN2D_Transformer(config.video_encoder, config.len_feature, config.num_classes, config.num_segments, config.fusion_type)
        
        elif model_name == "CNN3D": 
            from models.cnn3d.cnn3d import CNN3D
            shortcut_type = "B"
            sample_size = 224
            sample_duration=config.num_segments
            model_depth = 18
            mode = 'feature'
            return CNN3D(config.num_classes, shortcut_type, sample_size, sample_duration, model_depth, mode)
        
        elif model_name == "ViViT":
            from models.vivit.mm_vivit import MMViViT
            return MMViViT(config.video_encoder, config.len_feature, config.num_classes, config.num_segments, config.fusion_type)

        else:
            raise NotImplementedError("No such model")