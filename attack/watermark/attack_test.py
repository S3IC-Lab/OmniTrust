import torch
import json
import sys
from transformers import BertTokenizer, BertForMaskedLM
sys.path.append('..')
from attack.watermark.synonymsubstitution_attack import SynonymSubstitutionAttack
from attack.watermark.worddeletion_attack import WordDeletionAttack
from attack.watermark.contextaware_attack import ContextAwareSynonymSubstitutionAttack
from modules.watermark.auto_watermark import AutoWatermark
from utils.watermark.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load data
with open('../data/dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
item = json.loads(lines[0])
prompt = item['prompt']
natural_text = item['natural_text']


def test_algorithm(algorithm_name):
    # Check algorithm name
    assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'DIP', 'Unbiased',
                              'UPV', 'TS', 'SynthID', 'EXP', 'EXPGumbel', 'EXPEdit', 'ITSEdit']

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transformers config
    # use opt1.3b model as test model
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device),
                                             tokenizer=AutoTokenizer.from_pretrained("facebook/opt-1.3b"),
                                             vocab_size=50272,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=230,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)

    # Load watermark algorithm
    myWatermark = AutoWatermark.load(f'{algorithm_name}',
                                     algorithm_config=f'../config/watermark_config/{algorithm_name}.json',
                                     transformers_config=transformers_config, delta=1)

    watermarked_text = myWatermark.generate_watermarked_text(prompt)
    unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
    #print("The watermarked text: \n", watermarked_text)

    # Perform word deletion attack on the watermarked text
    attack = WordDeletionAttack(model=myWatermark, dataset=[watermarked_text], delete_ratio=0.5)
    worddeletion_attacked_data = attack.perform_attack()

    # Perform word synonymsubstitution attack on the watermarked text
    attack = SynonymSubstitutionAttack(model=myWatermark, dataset=[watermarked_text], ratio=0.8)
    synonymsubstitution_attacked_data = attack.perform_attack()

    # Perform word contextawaresynonymsubstitution attack on the watermarked text
    # 加载 BERT 模型和分词器
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # 创建攻击对象时传入参数
    #attack = ContextAwareSynonymSubstitutionAttack(model=myWatermark, dataset=[watermarked_text], ratio=0.5,tokenizer=tokenizer, bert_model=bert_model)
    contextawaresynonymsubstitution_attacked_data = attack.perform_attack()


    # Print the attacked (modified) watermarked text
    #print("\n The worddeletion_attacked watermarked text: \n", worddeletion_attacked_data[0])
    #print("\n The synonymsubstitution_attacked watermarked  text: \n", synonymsubstitution_attacked_data[0])
    #print("\n The contextawaresynonymsubstitution_attacked watermarked  text: \n", contextawaresynonymsubstitution_attacked_data[0])


    # Perform watermark detection on the original and attacked texts
    #print("\nThe detection result of watermarked text/unwatermarked text/natural text: ")

    detect_result = myWatermark.detect_watermark(watermarked_text)  # Check the watermarked text
    print("Detection result of watermarked text:", detect_result)

    detect_result = myWatermark.detect_watermark(worddeletion_attacked_data[0])  # Check the attacked text
    print("Detection result of deletion_attacked text:", detect_result)
    detect_result = myWatermark.detect_watermark(synonymsubstitution_attacked_data[0])  # Check the attacked text
    print("Detection result of synonymsubstitution_attacked text:", detect_result)
    #detect_result = myWatermark.detect_watermark(contextawaresynonymsubstitution_attacked_data[0])  # Check the attacked text
    #print("Detection result of contextawaresynonymsubstitution_attacked text:", detect_result)

    detect_result = myWatermark.detect_watermark(unwatermarked_text)  # Check the unwatermarked text
    print("Detection result of unwatermarked text:", detect_result)

    detect_result = myWatermark.detect_watermark(natural_text)  # Check the natural text
    print("Detection result of natural text:", detect_result)


if __name__ == '__main__':
    print("result of KGW:")
    test_algorithm('KGW')
    print("result of Unigram:")
    test_algorithm('Unigram')
    print("result of SWEET:")
    test_algorithm('SWEET')
    print("result of EWD:")
    test_algorithm('EWD')
    ##print("result of SIR:")
    #test_algorithm('SIR')
    print("result of DIP:")
    test_algorithm('DIP')
    print("result of Unbiased:")
    test_algorithm('Unbiased')
    print("result of SynthID:")
    test_algorithm('SynthID')
    print("result of EXP:")
    test_algorithm('EXP')






