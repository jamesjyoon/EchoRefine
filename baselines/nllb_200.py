import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


NLLB_LANG_CODES = {
    "eng": "eng_Latn",
    "npi": "npi_Deva",
    "ben": "ben_Beng",
    "sin": "sin_Sinh",
    "mya": "mya_Mymr",
    "kor": "kor_Hang",
    "tam": "tam_Taml",
    "hin": "hin_Deva",
    "fra": "fra_Latn",
}


class NLLB200Baseline:
    """NLLB-200 translation baseline for FLORES-style ISO-639-3 inputs."""

    def __init__(
        self,
        model_name="facebook/nllb-200-distilled-600M",
        device_map="auto",
        dtype=torch.float16,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )

    @staticmethod
    def lang_code(iso_639_3):
        try:
            return NLLB_LANG_CODES[iso_639_3]
        except KeyError as exc:
            raise ValueError(f"Unsupported NLLB language code: {iso_639_3}") from exc

    def translate_batch(
        self,
        source_texts,
        src_lang="eng",
        tgt_lang="npi",
        batch_size=8,
        max_length=256,
        num_beams=4,
    ):
        src_code = self.lang_code(src_lang)
        tgt_code = self.lang_code(tgt_lang)
        self.tokenizer.src_lang = src_code
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        outputs = []
        for start in range(0, len(source_texts), batch_size):
            batch = source_texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                )
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [text.strip() for text in outputs]

    def translate(self, source_texts, src_lang="eng", tgt_lang="npi", **kwargs):
        if isinstance(source_texts, str):
            return self.translate_batch([source_texts], src_lang, tgt_lang, **kwargs)[0]
        return self.translate_batch(source_texts, src_lang, tgt_lang, **kwargs)
