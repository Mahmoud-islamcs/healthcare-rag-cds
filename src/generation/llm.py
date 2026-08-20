import os
import re
import time
import logging
from typing import Optional, Any
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

class UnifiedLLM:
    def __init__(
        self,
        provider: str = "groq",
        model_id: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
        max_new_tokens: int = 2500,
        max_retries: int = 3,
        backoff_factor: float = 1.5
    ):
        self.provider = provider.lower()
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        if self.provider == "groq":
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                        api_key = st.secrets["GROQ_API_KEY"]
                except Exception:
                    pass

            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable is not set. "
                    "Please configure your GROQ_API_KEY in .env or Streamlit Secrets."
                )
            self.client = Groq(api_key=api_key)
            logger.info(f"Initialized Groq LLM client with model: {self.model_id}")

        elif self.provider in ["local", "transformers"]:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from src.utils.device import get_optimal_device

            self.device = get_optimal_device("auto")
            logger.info(f"Loading local model {self.model_id} on device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(self, system_prompt: Any, user_prompt: Any) -> str:
        # Defensively handle dict inputs
        if isinstance(user_prompt, dict):
            user_prompt = user_prompt.get("user_prompt", str(user_prompt))
        if isinstance(system_prompt, dict):
            system_prompt = system_prompt.get("system_prompt", str(system_prompt))
        
        system_str = str(system_prompt)
        user_str = str(user_prompt)

        if self.provider == "groq":
            raw_res = self._generate_groq_with_retry(system_str, user_str)
        elif self.provider in ["local", "transformers"]:
            raw_res = self._generate_local(system_str, user_str)
        else:
            return "LLM generation provider not configured."

        # Strip reasoning thinking tags and normalize
        cleaned = re.sub(r'<think>.*?</think>', '', raw_res, flags=re.DOTALL | re.IGNORECASE).strip()
        return cleaned

    def _generate_groq_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        models_to_try = [self.model_id]
        fallback_candidates = ["qwen/qwen3.6-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
        for fb in fallback_candidates:
            if fb not in models_to_try:
                models_to_try.append(fb)

        last_exception = None
        for current_model in models_to_try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=current_model,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                    )
                    choice = chat_completion.choices[0]
                    return choice.message.content or ""
                except Exception as e:
                    last_exception = e
                    err_msg = str(e)
                    is_rate_limit = "429" in err_msg or "rate_limit" in err_msg.lower()
                    logger.warning(
                        f"Groq API call attempt {attempt}/{self.max_retries} on model '{current_model}' failed: {e}"
                    )
                    if is_rate_limit and attempt == 1 and len(models_to_try) > 1:
                        # Immediately failover to next model without waiting out the long rate-limit backoff
                        logger.info(f"Failing over to next available Groq model due to rate limit on '{current_model}'")
                        break
                    if attempt < self.max_retries:
                        sleep_time = min(self.backoff_factor ** attempt, 2.0)
                        time.sleep(sleep_time)

        logger.error(f"All Groq API models and attempts failed: {last_exception}")
        raise RuntimeError(f"Groq generation failed after all attempts: {last_exception}")


    def _generate_local(self, system_prompt: str, user_prompt: str) -> str:
        import torch
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False if self.temperature == 0.0 else True
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

