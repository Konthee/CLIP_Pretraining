from .constants import TEXT_COLUMN, IMAGE_COLUMN


def dataset_tokenization(batch_input, processor):
    """
    Description: Tokenize `text` and `image` column with processor
    Args:
        batch_input: Batch columns contains `text` and `image`
        processor: CLIP processor
    Returns:
        batch_output: Batch columns contains `input_ids`, `attention_mask` 
            and `pixel_values`
    """
    # Tokenize text
    text_outputs = processor(batch_input['text'], padding=True,truncation=True)

    # Tokenize image
    image_outputs = processor(images=batch_input['image'])

    # Combine the outputs
    batch_output = {
        "input_ids": text_outputs["input_ids"],
        "attention_mask": text_outputs["attention_mask"],
        "pixel_values": image_outputs["pixel_values"]
    }


    return batch_output
