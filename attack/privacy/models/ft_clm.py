import transformers
import torch
import numpy as np
from heapq import nlargest
from .LLMBase import LLMBase


class FinetunedCasualLM(LLMBase):
    def __init__(self, model_path=None, arch=None, max_seq_len=1024, model=None, tokenizer=None):
        if ':' in model_path:
            model_path, self.model_revision = model_path.split(':')
        else:
            self.model_revision = 'main'
        if arch is None:
            self.arch = model_path
        else:
            self.arch = arch
        # default
        self.tokenizer_use_fast = True
        self.max_seq_len = max_seq_len
        self.verbose = True
        self._lm = model
        self._tokenizer = tokenizer
        super().__init__(model_path=model_path)
    
    @property
    def tokenizer(self):
        return self._tokenizer

    def load_local_model(self, model_path=None):
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._lm.config.pad_token_id = self._lm.config.eos_token_id
        
    def query(self, text, new_str_only=False):
        input_ids = self._tokenizer.encode(text, return_tensors='pt')

        # Implement the code to query the open-source model
        output = self._lm.generate(
            input_ids=input_ids.to('cuda'),
            max_new_tokens=self.max_seq_len,
            # do_sample=sampling_args.do_sample,
            return_dict_in_generate=True,
            
        )
        # Decode the generated text back to a readable string
        if new_str_only:
            generated_text = self._tokenizer.decode(output.sequences[0][len(input_ids[0]):], skip_special_tokens=True)
        else:
            generated_text = self._tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return generated_text
        
    def evaluate(self, text, tokenized=False):
        if tokenized:
            input_ids = text
        else:
            # Encode the text prompt and generate a response
            input_ids = self._tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=self.max_seq_len)
            # output = self.model.generate(input_ids)
        
        # Implement the code to query the open-source model
        input_ids = input_ids.to('cuda')
        output = self._lm(
            input_ids=input_ids,
            labels=input_ids.clone(),
        )
        return output.loss.item()
        
    def evaluate_ppl(self, text, tokenized=False):
        loss = self.evaluate(text, tokenized=tokenized)
        return np.exp(loss)

    def generate_neighbors(self, text, p=0.7, k=5, n=50):
        tokenized = self._tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_seq_len).input_ids.to('cuda')
        dropout = torch.nn.Dropout(p)

        seq_len = tokenized.shape[1]
        cand_scores = {}
        for target_index in range(1, seq_len):
            target_token = tokenized[0, target_index]
            
            # Embed the sequence
            if isinstance(self._lm, transformers.LlamaForCausalLM):
                embedding = self._lm.get_input_embeddings()(tokenized)
            elif isinstance(self._lm, transformers.GPT2LMHeadModel):
                embedding = self._lm.transformer.wte.weight[tokenized]
            else:
                raise RuntimeError(f'Unsupported model type for neighborhood generation: {type(self._lm)}')
            
            # Apply dropout only to the target token embedding in the sequence
            embedding = torch.cat([
                embedding[:, :target_index, :], 
                dropout(embedding[:, target_index:target_index+1, :]), 
                embedding[:, target_index+1:, :]
            ], dim=1)

            # Get model's predicted posterior distributions over all positions in the sequence
            probs = torch.softmax(self._lm(inputs_embeds=embedding).logits, dim=2)
            original_prob = probs[0, target_index, target_token].item()

            # Find the K most probable token replacements, not including the target token
            # Find top K+1 first because target could still appear as a candidate
            cand_probs, cands = torch.topk(probs[0, target_index, :], k + 1)
            
            # Score each candidate
            for prob, cand in zip(cand_probs, cands):
                if cand == target_token:
                    continue
                denominator = (1 - original_prob) if original_prob < 1 else 1E-6
                score = prob.item() / denominator
                cand_scores[(cand, target_index)] = score
        
        # Generate and return the neighborhood of sequences
        neighborhood = []
        top_keys = nlargest(n, cand_scores, key=cand_scores.get)
        for cand, index in top_keys:
            neighbor = torch.clone(tokenized)
            neighbor[0, index] = cand
            neighborhood.append(self._tokenizer.batch_decode(neighbor)[0])
        
        return neighborhood