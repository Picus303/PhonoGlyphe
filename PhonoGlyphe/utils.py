import torch.nn as nn

from .architecture import Transformer



def build_transformer(dropout: float,
                      source_vocab_size: int, target_vocab_size: int, context_length:int,
                      encoder_block_count: int, decoder_block_count: int,
                      encoder_self_attention_head_count: int, decoder_self_attention_head_count: int,
                      decoder_cross_attention_head_count: int,
                      encoder_self_attention_abstraction_coef: float, decoder_self_attention_abstraction_coef: float,
                      decoder_cross_attention_abstraction_coef: float,
                      encoder_feed_forward_abstraction_coef: float, decoder_feed_forward_abstraction_coef: float,
                      dim: int, epsilon: float) -> Transformer:
    """
    Build a transformer model with the given parameters.

    Args:
        dropout: The dropout rate to use in the model.
        source_vocab_size: The size of the source vocabulary.
        target_vocab_size: The size of the target vocabulary.
        context_length: The length of the context to use.
        encoder_block_count: The number of blocks in the encoder.
        decoder_block_count: The number of blocks in the decoder.
        encoder_self_attention_head_count: The number of heads in the encoder self-attention.
        decoder_self_attention_head_count: The number of heads in the decoder self-attention.
        decoder_cross_attention_head_count: The number of heads in the decoder cross-attention.
        encoder_self_attention_abstraction_coef: The ratio between the model dimension and the self-attention dimension in the encoder.
        decoder_self_attention_abstraction_coef: The ratio between the model dimension and the self-attention dimension in the decoder.
        decoder_cross_attention_abstraction_coef: The ratio between the model dimension and the cross-attention dimension in the decoder.
        encoder_feed_forward_abstraction_coef: The ratio between the model dimension and the feed-forward dimension in the encoder.
        decoder_feed_forward_abstraction_coef: The ratio between the model dimension and the feed-forward dimension in the decoder.
        dim: The dimension of the model.
        epsilon: The epsilon value to use in the normalisation layers.

    Returns:
        A transformer model with the given parameters.
    """

    # Arguments for the embeddings
    source_embeddings_params = (source_vocab_size, dim)
    target_embeddings_params = (target_vocab_size, dim)
    positional_embeddings_params = (context_length, dim)

    # Arguments for the encoder and decoder
    encoder_params = (encoder_block_count, ((dim, epsilon),                                                                                         # Normalisation layer
                                            (context_length, dim, encoder_self_attention_head_count, encoder_self_attention_abstraction_coef),      # Self-attention
                                            (dim, encoder_feed_forward_abstraction_coef)), (dim, epsilon))                                          # Feed-forward

    decoder_params = (decoder_block_count, ((dim, epsilon),                                                                                         # Normalisation layer
                                            (context_length, dim, decoder_self_attention_head_count, decoder_self_attention_abstraction_coef),      # Self-attention
                                            (context_length, dim, decoder_cross_attention_head_count, decoder_cross_attention_abstraction_coef),    # Cross-attention
                                            (dim, decoder_feed_forward_abstraction_coef)), (dim, epsilon))                                          # Feed-forward

    # Arguments for the projection layer
    projection_params = (context_length, dim, target_vocab_size)

    # Build the model
    transformer_model = Transformer(source_embeddings_params, target_embeddings_params,
                                    positional_embeddings_params,
                                    encoder_params, decoder_params,
                                    projection_params,
                                    dropout)

    # Initialize the weights
    for param in transformer_model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer_model