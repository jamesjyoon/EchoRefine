import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


LANGUAGE_NAMES = {
    "npi": "Nepali",
    "ben": "Bengali",
    "sin": "Sinhala",
    "mya": "Burmese",
    "kor": "Korean",
    "tam": "Tamil",
    "hin": "Hindi",
    "fra": "French",
}


class Aya23Baseline:
    """Aya-23 prompting baseline for multilingual translation."""

    def __init__(
        self,
        model_name="CohereLabs/aya-23-35B",
        device_map="auto",
        load_in_4bit=True,
        dtype=torch.float16,
        token=None,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            token=token,
        )

    @staticmethod
    def language_name(lang_iso):
        try:
            return LANGUAGE_NAMES[lang_iso]
        except KeyError as exc:
            raise ValueError(f"Unsupported Aya target language: {lang_iso}") from exc

    def format_prompt(self, source, tgt_lang):
        lang_name = self.language_name(tgt_lang)
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Return only the translation.",
            },
            {
                "role": "user",
                "content": f"Translate this English text into {lang_name}:\n{source}",
            },
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            "System: You are a professional translator. Return only the translation.\n"
            f"User: Translate this English text into {lang_name}:\n{source}\n"
            "Assistant:"
        )

    def translate_batch(
        self,
        source_texts,
        src_lang="eng",
        tgt_lang="npi",
        batch_size=4,
        max_new_tokens=256,
    ):
        if src_lang != "eng":
            raise ValueError("Aya23Baseline currently expects English source text.")

        outputs = []
        for start in range(0, len(source_texts), batch_size):
            batch = source_texts[start:start + batch_size]
            prompts = [self.format_prompt(source, tgt_lang) for source in batch]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            outputs.extend(text.strip() for text in decoded)
        return outputs

    def translate(self, source_texts, src_lang="eng", tgt_lang="npi", **kwargs):
        if isinstance(source_texts, str):
            return self.translate_batch([source_texts], src_lang, tgt_lang, **kwargs)[0]
        return self.translate_batch(source_texts, src_lang, tgt_lang, **kwargs)
