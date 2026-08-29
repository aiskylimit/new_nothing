"""Token extraction helpers for SEGDLoss (vision/text split, teacher vs student padding)."""

import torch

IMAGE_TOKEN_ID_MIN = 151643
IMAGE_TOKEN_ID_MAX = 151656
STUDENT_IMAGE_TOKEN_INDEX = -200


def extract_text_hidden_states(
    hidden_states,
    sample_idx,
    num_text_tokens,
    num_vision_tokens,
    is_teacher=False,
    has_image=True,
):
    text_hidden_list = []
    for layer_hidden in hidden_states:
        if has_image:
            if is_teacher:
                text_hidden = layer_hidden[sample_idx, -num_text_tokens:, :]
            else:
                text_hidden = layer_hidden[
                    sample_idx, num_vision_tokens : (num_vision_tokens + num_text_tokens), :
                ]
        else:
            if is_teacher:
                text_hidden = layer_hidden[sample_idx, -num_text_tokens:, :]
            else:
                text_hidden = layer_hidden[sample_idx, :num_text_tokens, :]
        text_hidden_list.append(text_hidden)
    return text_hidden_list


def extract_vision_hidden_states(
    hidden_states,
    sample_idx,
    num_vision_tokens,
    num_text_tokens,
    is_teacher=False,
):
    vision_hidden_list = []
    for layer_hidden in hidden_states:
        if is_teacher:
            start_idx = -(num_vision_tokens + num_text_tokens)
            end_idx = -num_text_tokens if num_text_tokens > 0 else None
            vision_hidden = layer_hidden[sample_idx, start_idx:end_idx, :]
        else:
            vision_hidden = layer_hidden[sample_idx, :num_vision_tokens, :]
        vision_hidden_list.append(vision_hidden)
    return vision_hidden_list


def count_text_tokens_teacher(input_ids_row):
    mask = (input_ids_row < IMAGE_TOKEN_ID_MIN) | (input_ids_row > IMAGE_TOKEN_ID_MAX)
    return int(mask.sum().item())


def count_text_tokens_student(input_ids_row):
    mask = (input_ids_row < IMAGE_TOKEN_ID_MIN) | (input_ids_row > IMAGE_TOKEN_ID_MAX)
    mask = mask & (input_ids_row != STUDENT_IMAGE_TOKEN_INDEX)
    return int(mask.sum().item())
