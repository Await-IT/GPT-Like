import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import math
from collections import Counter, defaultdict
import re
import json
import regex
import tokenize
import io
from keyword import iskeyword

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    vocab_size = 50000
    d_model = 256
    n_head = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 1024
    batch_size = 16
    learning_rate = 3e-4
    pretrain_epochs = 30
    finetune_epochs = 30
    pretrain_file = r"C:\Users\Ваше имя\OneDrive\Desktop\pretrain.txt"
    finetune_file = r"C:\Users\Ваше имя\OneDrive\Desktop\fiten.txt"
    model_save_path = r"C:\Users\Ваше имя\OneDrive\Desktop\LLM.bin"
    tokenizer_path = r"C:\Users\Ваше имя\OneDrive\Desktop\tokenizer.json"
    
    gradient_accumulation_steps = 4
    
    # ==================== СПЕЦИАЛЬНЫЕ ТОКЕНЫ ====================
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token = "<pad>"
    unk_token = "<unk>"
    
    user_token = "<user>"
    bot_token = "<bot>"
    start_token = "<start>"
    end_token = "<end>"
    
    paragraph_start_token = "<p>"
    paragraph_end_token = "</p>"
    
    text_start_token = "<text>"
    text_end_token = "</text>"
    
    book_start_token = "<book>"
    book_end_token = "</book>"
    
    code_start_token = "<code>"
    code_end_token = "</code>"
    
    # ==================== СПЕЦИАЛЬНЫЕ ТОКЕНЫ ДЛЯ КОДА ====================
    code_class_token = "<class>"
    code_def_token = "<def>"
    code_if_token = "<if>"
    code_for_token = "<for>"
    code_while_token = "<while>"
    code_try_token = "<try>"
    code_import_token = "<import>"
    code_from_token = "<from>"
    code_return_token = "<return>"
    code_self_token = "<self>"
    code_none_token = "<none>"
    code_true_token = "<true>"
    code_false_token = "<false>"
    
    code_indent_token = "<indent>"
    code_dedent_token = "<dedent>"
    code_newline_token = "<newline>"
    code_space_token = "<space>"
    code_comment_token = "<comment>"
    code_string_token = "<string>"
    code_number_token = "<number>"
    code_variable_token = "<var>"
    code_function_token = "<func>"
    code_argument_token = "<arg>"
    code_keyword_token = "<kw>"
    code_operator_token = "<op>"
    code_delimiter_token = "<delim>"
    
    special_tokens = [
        bos_token, eos_token, pad_token, unk_token,
        user_token, bot_token, start_token, end_token,
        paragraph_start_token, paragraph_end_token,
        text_start_token, text_end_token,
        book_start_token, book_end_token,
        code_start_token, code_end_token,
        code_class_token, code_def_token, code_if_token, code_for_token,
        code_while_token, code_try_token, code_import_token, code_from_token,
        code_return_token, code_self_token, code_none_token, code_true_token,
        code_false_token, code_indent_token, code_dedent_token, code_newline_token,
        code_space_token, code_comment_token, code_string_token, code_number_token,
        code_variable_token, code_function_token, code_argument_token,
        code_keyword_token, code_operator_token, code_delimiter_token
    ]
    
    @classmethod
    def to_dict(cls):
        return {
            'vocab_size': cls.vocab_size,
            'd_model': cls.d_model,
            'n_head': cls.n_head,
            'num_layers': cls.num_layers,
            'd_ff': cls.d_ff,
            'dropout': cls.dropout,
            'max_seq_len': cls.max_seq_len,
            'batch_size': cls.batch_size,
            'learning_rate': cls.learning_rate,
            'pretrain_epochs': cls.pretrain_epochs,
            'finetune_epochs': cls.finetune_epochs,
            'gradient_accumulation_steps': cls.gradient_accumulation_steps,
            'pretrain_file': cls.pretrain_file,
            'finetune_file': cls.finetune_file,
            'model_save_path': cls.model_save_path,
            'tokenizer_path': cls.tokenizer_path,
            'special_tokens': cls.special_tokens
        }

# ==================== СПЕЦИАЛЬНЫЙ ТОКЕНИЗАТОР ДЛЯ PYTHON ====================
class CodeTokenizer:
    PYTHON_KEYWORDS = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
        'try', 'while', 'with', 'yield'
    }
    
    BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
        'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
        'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
        'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
        'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview',
        'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property',
        'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice',
        'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
    }
    
    @staticmethod
    def tokenize_python_code(code):
        tokens = []
        indent_level = 0
        
        try:
            token_stream = tokenize.generate_tokens(io.StringIO(code).readline)
            
            for tok in token_stream:
                token_type = tokenize.tok_name[tok.type]
                token_value = tok.string
                
                if token_type == 'INDENT':
                    indent_level += 1
                    tokens.append(Config.code_indent_token)
                    continue
                elif token_type == 'DEDENT':
                    indent_level -= 1
                    tokens.append(Config.code_dedent_token)
                    continue
                elif token_type == 'NEWLINE':
                    tokens.append(Config.code_newline_token)
                    continue
                
                if token_type == 'NAME':
                    if token_value in CodeTokenizer.PYTHON_KEYWORDS:
                        if token_value == 'class':
                            tokens.append(Config.code_class_token)
                        elif token_value == 'def':
                            tokens.append(Config.code_def_token)
                        elif token_value == 'if':
                            tokens.append(Config.code_if_token)
                        elif token_value == 'for':
                            tokens.append(Config.code_for_token)
                        elif token_value == 'while':
                            tokens.append(Config.code_while_token)
                        elif token_value == 'try':
                            tokens.append(Config.code_try_token)
                        elif token_value == 'import':
                            tokens.append(Config.code_import_token)
                        elif token_value == 'from':
                            tokens.append(Config.code_from_token)
                        elif token_value == 'return':
                            tokens.append(Config.code_return_token)
                        elif token_value == 'self':
                            tokens.append(Config.code_self_token)
                        elif token_value in ('None', 'True', 'False'):
                            if hasattr(Config, f'code_{token_value.lower()}_token'):
                                tokens.append(getattr(Config, f'code_{token_value.lower()}_token'))
                            else:
                                tokens.append(token_value)
                        elif iskeyword(token_value):
                            tokens.append(Config.code_keyword_token)
                        else:
                            if token_value in CodeTokenizer.BUILTINS:
                                tokens.append(Config.code_function_token)
                            else:
                                if token_value.startswith('_'):
                                    tokens.append(Config.code_variable_token)
                                elif token_value.isupper():
                                    tokens.append(Config.code_variable_token)
                                else:
                                    if re.match(r'^[a-z_][a-z0-9_]*$', token_value):
                                        tokens.append(Config.code_variable_token)
                                    elif re.match(r'^[A-Z][a-zA-Z0-9]*$', token_value):
                                        tokens.append(Config.code_class_token)
                                    else:
                                        tokens.append(token_value)
                    else:
                        tokens.append(token_value)
                
                elif token_type == 'STRING':
                    tokens.append(Config.code_string_token)
                
                elif token_type == 'NUMBER':
                    tokens.append(Config.code_number_token)
                
                elif token_type == 'COMMENT':
                    tokens.append(Config.code_comment_token)
                
                elif token_type == 'OP':
                    if token_value in ('+', '-', '*', '/', '//', '%', '**', 
                                      '=', '==', '!=', '<', '>', '<=', '>=',
                                      '+=', '-=', '*=', '/=', '//=', '%=', '**=',
                                      '&', '|', '^', '~', '<<', '>>', '<<=', '>>=',
                                      'and', 'or', 'not', 'is', 'in'):
                        tokens.append(Config.code_operator_token)
                    elif token_value in ('(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '@'):
                        tokens.append(Config.code_delimiter_token)
                    else:
                        tokens.append(token_value)
                
                elif token_type == 'ENCODING' or token_type == 'ENDMARKER':
                    continue
                
                else:
                    tokens.append(token_value)
        
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return CodeTokenizer.fallback_tokenize(code)
        
        return tokens
    
    @staticmethod
    def fallback_tokenize(code):
        tokens = []
        lines = code.split('\n')
        
        for line in lines:
            stripped = line.lstrip()
            indent_count = len(line) - len(stripped)
            if indent_count > 0:
                tokens.append(Config.code_indent_token)
            
            line_tokens = stripped.split()
            for token in line_tokens:
                if token.startswith('#'):
                    tokens.append(Config.code_comment_token)
                elif token.startswith('"') or token.startswith("'") or token.startswith('"""') or token.startswith("'''"):
                    tokens.append(Config.code_string_token)
                elif token.isdigit() or (token.replace('.', '', 1).isdigit() and token.count('.') < 2):
                    tokens.append(Config.code_number_token)
                elif token in CodeTokenizer.PYTHON_KEYWORDS:
                    if token == 'class':
                        tokens.append(Config.code_class_token)
                    elif token == 'def':
                        tokens.append(Config.code_def_token)
                    elif token == 'if':
                        tokens.append(Config.code_if_token)
                    elif token == 'for':
                        tokens.append(Config.code_for_token)
                    elif token == 'while':
                        tokens.append(Config.code_while_token)
                    elif token == 'import':
                        tokens.append(Config.code_import_token)
                    elif token == 'from':
                        tokens.append(Config.code_from_token)
                    elif token == 'return':
                        tokens.append(Config.code_return_token)
                    elif token in ('None', 'True', 'False'):
                        if hasattr(Config, f'code_{token.lower()}_token'):
                            tokens.append(getattr(Config, f'code_{token.lower()}_token'))
                        else:
                            tokens.append(token)
                    else:
                        tokens.append(Config.code_keyword_token)
                elif token in CodeTokenizer.BUILTINS:
                    tokens.append(Config.code_function_token)
                elif re.match(r'^[a-z_][a-z0-9_]*$', token):
                    tokens.append(Config.code_variable_token)
                elif re.match(r'^[A-Z][a-zA-Z0-9]*$', token):
                    tokens.append(Config.code_class_token)
                else:
                    tokens.append(token)
            
            tokens.append(Config.code_newline_token)
        
        return tokens

# ==================== BPE ТОКЕНИЗАТОР ====================
class ImprovedBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.inverse_vocab = {}
        self.special_tokens = {}
        self.code_tokenizer = CodeTokenizer()
        
        self.regex_pattern = r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        
        self.text_block_pattern = r'<text>.*?</text>'
        self.book_block_pattern = r'<book>.*?</book>'
        self.code_block_pattern = r'<code>.*?</code>'
    
    def train(self, text_file, vocab_size=50000):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chars = sorted(list(set(text)))
        self.vocab = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)
        
        for i, token in enumerate(Config.special_tokens):
            token_id = len(self.vocab) + i
            self.vocab[token_id] = token
            self.special_tokens[token] = token_id
        
        processed_text_blocks = []
        
        code_blocks = re.findall(self.code_block_pattern, text, re.DOTALL)
        text_without_code = re.sub(self.code_block_pattern, ' __CODE_BLOCK__ ', text)
        
        book_blocks = re.findall(self.book_block_pattern, text_without_code, re.DOTALL)
        text_without_book = re.sub(self.book_block_pattern, ' __BOOK_BLOCK__ ', text_without_code)
        
        text_blocks = re.findall(self.text_block_pattern, text_without_book, re.DOTALL)
        plain_text = re.sub(self.text_block_pattern, ' __TEXT_BLOCK__ ', text_without_book)
        
        for match in regex.finditer(self.regex_pattern, plain_text):
            word = match.group()
            if word.strip() and word != '__CODE_BLOCK__' and word != '__BOOK_BLOCK__' and word != '__TEXT_BLOCK__':
                processed_text_blocks.append(list(word))
        
        for block in text_blocks:
            content = block.replace('<text>', '').replace('</text>', '').strip()
            if content:
                processed_text_blocks.append(list(content))
        
        for block in book_blocks:
            content = block.replace('<book>', '').replace('</book>', '').strip()
            if content:
                processed_text_blocks.append(list(content))
        
        for block in code_blocks:
            content = block.replace('<code>', '').replace('</code>', '').strip()
            if content:
                code_tokens = self.code_tokenizer.tokenize_python_code(content)
                if code_tokens:
                    processed_text_blocks.append(code_tokens)
        
        merge_rounds = vocab_size - self.vocab_size
        for round_num in range(merge_rounds):
            if round_num % 100 == 0:
                print(f"BPE round {round_num}/{merge_rounds}")
                
            pairs = Counter()
            for token_list in processed_text_blocks:
                for i in range(len(token_list) - 1):
                    pairs[(token_list[i], token_list[i+1])] += 1
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            new_token_id = self.vocab_size
            self.vocab[new_token_id] = new_token
            self.merges[best_pair] = new_token_id
            self.vocab_size += 1
            
            new_tokens = []
            for token_list in processed_text_blocks:
                i = 0
                new_list = []
                while i < len(token_list):
                    if i < len(token_list) - 1 and (token_list[i], token_list[i+1]) == best_pair:
                        new_list.append(new_token)
                        i += 2
                    else:
                        new_list.append(token_list[i])
                        i += 1
                new_tokens.append(new_list)
            processed_text_blocks = new_tokens
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.save(Config.tokenizer_path)
        print(f"Tokenizer trained with {self.vocab_size} tokens")
    
    def save(self, filepath):
        merges_serializable = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        
        data = {
            'vocab': self.vocab,
            'merges': merges_serializable,
            'special_tokens': self.special_tokens,
            'inverse_vocab': self.inverse_vocab
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = {int(k): v for k, v in data['vocab'].items()}
        self.merges = {}
        for k, v in data['merges'].items():
            parts = k.split(',')
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = v
        
        self.special_tokens = data['special_tokens']
        self.inverse_vocab = {k: int(v) for k, v in data['inverse_vocab'].items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text):
        tokens = []
        
        for special_token in [Config.code_start_token, Config.code_end_token,
                            Config.book_start_token, Config.book_end_token,
                            Config.text_start_token, Config.text_end_token]:
            if special_token in text:
                return self.encode_special_blocks(text)
        
        for match in regex.finditer(self.regex_pattern, text):
            word = match.group()
            if word in self.special_tokens:
                tokens.append(self.special_tokens[word])
                continue
                
            word_tokens = self.bpe_encode_word(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode_special_blocks(self, text):
        tokens = []
        
        parts = re.split(f'({Config.code_start_token}|{Config.code_end_token}|'
                        f'{Config.book_start_token}|{Config.book_end_token}|'
                        f'{Config.text_start_token}|{Config.text_end_token})', text)
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            if part in [Config.code_start_token, Config.book_start_token, Config.text_start_token]:
                block_type = part
                i += 1
                
                content_parts = []
                while i < len(parts) and parts[i] not in [Config.code_end_token, 
                                                         Config.book_end_token, 
                                                         Config.text_end_token]:
                    content_parts.append(parts[i])
                    i += 1
                
                content = ''.join(content_parts).strip()
                tokens.append(self.special_tokens[block_type])
                
                if block_type == Config.code_start_token:
                    code_tokens = self.code_tokenizer.tokenize_python_code(content)
                    for token in code_tokens:
                        if token in self.special_tokens:
                            tokens.append(self.special_tokens[token])
                        else:
                            word_tokens = self.bpe_encode_word(token)
                            tokens.extend(word_tokens)
                else:
                    for match in regex.finditer(self.regex_pattern, content):
                        word = match.group()
                        if word.strip():
                            if word in self.special_tokens:
                                tokens.append(self.special_tokens[word])
                            else:
                                word_tokens = self.bpe_encode_word(word)
                                tokens.extend(word_tokens)
                
                if i < len(parts) and parts[i] in [Config.code_end_token, 
                                                  Config.book_end_token, 
                                                  Config.text_end_token]:
                    tokens.append(self.special_tokens[parts[i]])
                    i += 1
            else:
                if part.strip():
                    for match in regex.finditer(self.regex_pattern, part):
                        word = match.group()
                        if word.strip():
                            if word in self.special_tokens:
                                tokens.append(self.special_tokens[word])
                            else:
                                word_tokens = self.bpe_encode_word(word)
                                tokens.extend(word_tokens)
                i += 1
        
        return tokens
    
    def bpe_encode_word(self, word):
        if word in self.inverse_vocab:
            return [self.inverse_vocab[word]]
            
        tokens = list(word)
        while len(tokens) > 1:
            min_rank = float('inf')
            best_pair = None
            best_idx = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges and self.merges[pair] < min_rank:
                    min_rank = self.merges[pair]
                    best_pair = pair
                    best_idx = i
            
            if best_pair is None:
                break
                
            new_token = best_pair[0] + best_pair[1]
            tokens = tokens[:best_idx] + [new_token] + tokens[best_idx+2:]
        
        result = []
        for token in tokens:
            if token in self.inverse_vocab:
                result.append(self.inverse_vocab[token])
            else:
                sub_tokens = self.bpe_encode_word(token)
                result.extend(sub_tokens)
        return result
    
    def decode(self, tokens):
        text = []
        i = 0
        
        while i < len(tokens):
            token_id = tokens[i]
            
            if token_id in self.vocab:
                token = self.vocab[token_id]
                
                if token == Config.code_indent_token:
                    text.append('    ')
                elif token == Config.code_dedent_token:
                    if text and text[-1] == '    ':
                        text.pop()
                elif token == Config.code_newline_token:
                    text.append('\n')
                elif token == Config.code_space_token:
                    text.append(' ')
                else:
                    text.append(token)
            else:
                text.append(Config.unk_token)
            
            i += 1
        
        return ''.join(text)
    
    def token_to_id(self, token):
        if token in self.inverse_vocab:
            return self.inverse_vocab[token]
        elif token in self.special_tokens:
            return self.special_tokens[token]
        else:
            return self.special_tokens[Config.unk_token]
    
    def id_to_token(self, token_id):
        return self.vocab.get(token_id, Config.unk_token)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_special_tokens(self):
        return self.special_tokens

# ==================== МОДЕЛЬ TRANSFORMER ====================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=8, num_layers=6, d_ff=2048, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.max_seq_len = max_seq_len
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        
        embeddings = self.dropout(token_embeddings + position_embeddings)
        
        attention_mask = self.generate_attention_mask(seq_len, input_ids.device)
        
        transformer_output = self.transformer(embeddings, mask=attention_mask)
        
        output = self.ln_final(transformer_output)
        logits = self.lm_head(output)
        
        return logits
    
    def generate_attention_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def compute_loss(self, logits, targets):
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        loss = nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=tokenizer.token_to_id(Config.pad_token))
        return loss

# ==================== ДАТАЛОАДЕР С ОПТИМИЗАЦИЯМИ ====================
class DialogDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length, is_pretrain=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_pretrain = is_pretrain
        self.dialogs = self.load_dialogs(file_path)
        
    def load_dialogs(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        dialogs = []
        
        if self.is_pretrain:
            raw_blocks = text.split('\n\n')
            
            for block in raw_blocks:
                block = block.strip()
                if not block:
                    continue
                
                if block.startswith('<code>') and block.endswith('</code>'):
                    dialogs.append([f"{Config.code_start_token} {block} {Config.code_end_token}"])
                elif block.startswith('<book>') and block.endswith('</book>'):
                    dialogs.append([f"{Config.book_start_token} {block} {Config.book_end_token}"])
                elif block.startswith('<text>') and block.endswith('</text>'):
                    dialogs.append([f"{Config.text_start_token} {block} {Config.text_end_token}"])
                else:
                    dialogs.append([f"{Config.text_start_token} {block} {Config.text_end_token}"])
        else:
            pattern = r'<p>\s*(.*?)\s*</p>'
            matches = re.findall(pattern, text, re.DOTALL)
            
            for match in matches:
                dialog_lines = []
                lines = match.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('<user>'):
                        dialog_lines.append(f"{Config.user_token} {line.replace('<user>', '').strip()}")
                    elif line.startswith('<bot>'):
                        bot_line = line.replace('<bot>', '').strip()
                        if '<start>' in bot_line and '<end>' in bot_line:
                            bot_line = bot_line.replace('<start>', f' {Config.start_token} ').replace('<end>', f' {Config.end_token} ')
                            dialog_lines.append(f"{Config.bot_token} {bot_line}")
                        else:
                            dialog_lines.append(f"{Config.bot_token} {Config.start_token} {bot_line} {Config.end_token}")
                
                if dialog_lines:
                    dialogs.append(dialog_lines)
        
        print(f"Loaded {len(dialogs)} dialogs from {file_path}")
        return dialogs
    
    def __len__(self):
        return len(self.dialogs)
    
    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        
        if self.is_pretrain:
            if len(dialog) == 1:
                full_text = f"{Config.bos_token} {dialog[0]} {Config.eos_token}"
            else:
                full_text = f"{Config.bos_token} {Config.paragraph_start_token} " + " ".join(dialog) + f" {Config.paragraph_end_token} {Config.eos_token}"
        else:
            full_text = f"{Config.bos_token} {Config.paragraph_start_token} " + " ".join(dialog) + f" {Config.paragraph_end_token} {Config.eos_token}"
        
        tokens = self.tokenizer.encode(full_text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return torch.tensor(input_ids), torch.tensor(target_ids)

# ==================== ОПТИМИЗИРОВАННЫЙ ДАТАЛОАДЕР ====================
class FastDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch = []
        for idx in self.indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        
        if batch:
            yield self.collate_fn(batch)
    
    @staticmethod
    def collate_fn(batch):
        input_ids, target_ids = zip(*batch)
        
        max_len = max(len(x) for x in input_ids)
        
        pad_id = tokenizer.token_to_id(Config.pad_token)
        batch_inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        batch_targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        
        for i, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
            batch_inputs[i, :len(inp)] = inp
            batch_targets[i, :len(tgt)] = tgt
        
        return batch_inputs, batch_targets

# ==================== ОПТИМИЗАЦИИ ДЛЯ БЫСТРОГО ОБУЧЕНИЯ ====================
def enable_fast_training():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
    else:
        scaler = None
        use_amp = False
    
    dataloader_config = {
        'num_workers': min(4, os.cpu_count() or 1),
        'pin_memory': True,
        'persistent_workers': True if os.cpu_count() and os.cpu_count() > 1 else False,
        'prefetch_factor': 2 if os.cpu_count() and os.cpu_count() > 1 else None,
        'batch_size': Config.batch_size
    }
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    return scaler, use_amp, dataloader_config

def fast_train_step(model, batch, optimizer, scaler=None, use_amp=False):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    
    optimizer.zero_grad(set_to_none=True)
    
    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = model.compute_loss(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        loss = model.compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
    
    return loss.item()

def train_fast(model, train_loader, val_loader, epochs, model_name="model"):
    scaler, use_amp, dl_config = enable_fast_training()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    print(f"Starting fast training on {device}...")
    print(f"Using mixed precision: {use_amp}")
    print(f"Batch size: {Config.batch_size}")
    
    torch.cuda.empty_cache()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            loss = fast_train_step(model, batch, optimizer, scaler, use_amp)
            train_loss += loss
            batch_count += 1
            
            scheduler.step()
            
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * Config.batch_size / elapsed
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss:.4f}, "
                      f"Speed={samples_per_sec:.1f} samples/sec, "
                      f"LR={scheduler.get_last_lr()[0]:.6f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = model.compute_loss(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = model.compute_loss(outputs, targets)
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss / batch_count
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Time={epoch_time:.1f}s")
        
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f"{model_name}_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved: {model_name}_epoch_{epoch+1}.pt")
    
    return model

def create_fast_dataloaders(tokenizer, pretrain=True):
    file_path = Config.pretrain_file if pretrain else Config.finetune_file
    dataset = DialogDataset(file_path, tokenizer, Config.max_seq_len, is_pretrain=pretrain)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = FastDataLoader(train_dataset, Config.batch_size, shuffle=True)
    val_loader = FastDataLoader(val_dataset, Config.batch_size, shuffle=False)
    
    print(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    return train_loader, val_loader

# ==================== ГЕНЕРАЦИЯ ТЕКСТА ====================
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50):
    model.eval()
    
    input_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_tokens], device=device)
    
    generated_tokens = input_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            if len(generated_tokens) >= Config.max_seq_len:
                break
                
            inputs = torch.tensor([generated_tokens], device=device)
            
            outputs = model(inputs)
            next_token_logits = outputs[0, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == tokenizer.token_to_id(Config.eos_token):
                break
                
            generated_tokens.append(next_token)
    
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

# ==================== ГЛАВНАЯ ПРОГРАММА ====================
def main():
    global device, tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if hasattr(os, 'sched_setscheduler'):
        try:
            os.nice(-10)
        except:
            pass
    
    tokenizer = ImprovedBPETokenizer()
    
    if not os.path.exists(Config.tokenizer_path):
        print("Training tokenizer with fast mode...")
        tokenizer.train(Config.pretrain_file, Config.vocab_size)
    else:
        print("Loading existing tokenizer...")
        tokenizer.load(Config.tokenizer_path)
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    test_code = '''<code>
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
</code>'''
    
    print("\nTesting tokenization...")
    tokens = tokenizer.encode(test_code)
    print(f"Tokenized {len(tokens)} tokens")
    
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded preview:\n{decoded[:200]}...")
    
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"\nInitializing model with vocab_size={vocab_size}...")
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_head=Config.n_head,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.max_seq_len
    )
    
    model.to(device)
    print(f"Model initialized on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nCreating fast dataloaders for pretraining...")
    train_loader, val_loader = create_fast_dataloaders(tokenizer, pretrain=True)
    
    print("\nStarting fast pretraining...")
    model = train_fast(model, train_loader, val_loader, Config.pretrain_epochs, "pretrain")
    
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"Model saved to {Config.model_save_path}")
    
    print("\nTesting text generation...")
    test_prompts = [
        f"{Config.user_token} Напиши функцию для вычисления факториала",
        f"{Config.user_token} Расскажи о трансформерах в машинном обучении",
        f"{Config.code_start_token} def calculate_sum(numbers):"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        generated = generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7)
        print(f"Generated:\n{generated}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()