import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import argparse
from translate import Translator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, LlamaTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM

import sys
sys.path.append('..')
from data.data_registry.c4 import C4Dataset
from data.data_registry.HumanEval import HumanEvalDataset
from data.data_registry.wmt16de_en import WMT16DE_ENDataset
from data.data_registry.cnn_dailymail import CNN_DailyMailDataset
from modules.watermark.auto_watermark import AutoWatermark
from eval.watermark_success_rate_calculator import DynamicThresholdSuccessRateCalculator
from eval.watermark.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from eval.watermark_text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger, GPTTextDiscriminator, RougeCalculator, MauveCalculator, BERTScoreCalculator
from eval.watermark.quality_analysis import (DirectTextQualityAnalysisPipeline, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline, QualityPipelineReturnType)
from utils.watermark.transformers_config import TransformersConfig
from utils.watermark.visualize.font_settings import FontSettings
from utils.watermark.visualize.visualizer import DiscreteVisualizer, ContinuousVisualizer, MultiLevelDiscreteVisualizer
from utils.watermark.visualize.legend_settings import DiscreteLegendSettings, ContinuousLegendSettings
from utils.watermark.visualize.page_layout_settings import PageLayoutSettings
from utils.watermark.visualize.color_scheme import ColorSchemeForDiscreteVisualization, ColorSchemeForContinuousVisualization, ColorSchemeForMultiLevelVisualization, ColorSchemeForITSEdit
from utils.watermark.tools.text_editor import TruncatePromptTextEditor, TruncateTaskTextEditor, WordDeletion, TypoAttack, MispellingAttack, ExpansionAttack, ContractionAttack, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor, CodeGenerationTextEditor

 # Device
device = "cuda" if torch.cuda.is_available() else "cpu"

def test_add_watermark_algorithm(algorithm_name, model_path, visualize_or_not):
    # Check algorithm name
    assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'DIP', 'Unbiased', 
                              'UPV', 'TS', 'SynthID', 'EXP', 'EXPGumbel', 'EXPEdit', 'ITSEdit', 'PF', 'MorphMark']

    # Currently supported datasets: C4, Humaneval, WMT16_de_en, CNN_DailyMail
    my_dataset = C4Dataset('../data/dataset/c4/c4.json')
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # If the size of the vocabulary is inconsistent, adjust the model's vocab_size.
    if model.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
    else:
        pass

    # Transformers config
    transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=tokenizer.vocab_size,
                                            device=device,
                                            max_new_tokens=200,
                                            min_length=230,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)
                                            
    # Load watermark algorithm
    myWatermark = AutoWatermark.load(f'{algorithm_name}', 
                                     algorithm_config=f'../config/watermark_config/{algorithm_name}.json',
                                     transformers_config=transformers_config)

    # Generate text: Take the first piece of data for testing.
    watermarked_text = myWatermark.generate_watermarked_text(my_dataset.get_prompt(0))
    unwatermarked_text = myWatermark.generate_unwatermarked_text(my_dataset.get_prompt(0))
    print("The watermarked text: \n", watermarked_text)

    '''
    # sample detection {is_watermark_or_not, z-score}
    print("\n The detection result of watermarked/unwatermarked/natural text: ")
    detect_result = myWatermark.detect_watermark(watermarked_text)
    print(detect_result)
    detect_result = myWatermark.detect_watermark(unwatermarked_text)
    print(detect_result)
    detect_result = myWatermark.detect_watermark(natural_text)
    print(detect_result)
    '''

    # Generate visualized image or not
    if visualize_or_not == True:
        # Exclude algorithms that do not support visualization
        supported_algorithms = ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'DIP', 'Unbiased', 'UPV', 'TS', 'EXP', 'EXPEdit', 'EXPGumbel', 'SynthID',  'PF', 'ITSEdit']
        try:
            if algorithm_name not in supported_algorithms:
                raise ValueError(f"\n============ {algorithm_name} algorithm currently does not support visualization now, coming soon~ ============\n")
        except ValueError as e:
            print(e)
            sys.exit(1)

        # Get data for visualization
        watermarked_data = myWatermark.get_data_for_visualization(watermarked_text)
        unwatermarked_data = myWatermark.get_data_for_visualization(unwatermarked_text)

        # Visualizer setting
        if algorithm_name in ['KGW','Unigram','SWEET','EWD','SIR','XSIR','DIP','Unbiased','UPV','TS','ITSEdit']:
            
            # Init color scheme
            if algorithm_name == 'ITSEdit':
                color_scheme = ColorSchemeForITSEdit()
            else:
                color_scheme = ColorSchemeForDiscreteVisualization()

            # Init visualizer for KGW family & ITSEdit
            visualizer = DiscreteVisualizer(color_scheme=color_scheme,
                                            font_settings=FontSettings(),
                                            page_layout_settings=PageLayoutSettings(),
                                            legend_settings=DiscreteLegendSettings())

        elif algorithm_name in ['EXP','EXPEdit','EXPGumbel','PF']:
            # Init visualizer for Christ family
            visualizer = ContinuousVisualizer(color_scheme=ColorSchemeForContinuousVisualization(),
                                              font_settings=FontSettings(), 
                                              page_layout_settings=PageLayoutSettings(),
                                              legend_settings=ContinuousLegendSettings())

        elif algorithm_name == 'SynthID':
            # Init visualizer for SynthID
            visualizer = MultiLevelDiscreteVisualizer(color_scheme=ColorSchemeForMultiLevelVisualization(),
                                                      font_settings=FontSettings(),
                                                      page_layout_settings=PageLayoutSettings(),
                                                      legend_settings=DiscreteLegendSettings())

        # Visualize
        watermarked_img = visualizer.visualize(data=watermarked_data,
                                                show_text=True,
                                                visualize_weight=True,
                                                display_legend=True)

        unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                                    show_text=True,
                                                    visualize_weight=True,
                                                    display_legend=True)
        # Save visualized image
        watermarked_img.save(f"{algorithm_name}'s watermarked.png")
        unwatermarked_img.save(f"{algorithm_name}'s unwatermarked.png")
        print(f"\n============ The visualized images have been saved successfully. ============\n")

    else:
        pass


def test_detect_algorithm(algorithm_name, model_path, attack_methods, labels, rules, target_fpr):
    my_dataset = C4Dataset('../data/dataset/c4/c4.json')
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained(model_path).to(device),
                                             tokenizer=AutoTokenizer.from_pretrained(model_path),
                                             vocab_size=50272,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=230,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    myWatermark = AutoWatermark.load(f'{algorithm_name}',
                                     algorithm_config=f'../config/watermark_config/{algorithm_name}.json',
                                     transformers_config=transformers_config)

    print("\nApplying attacks:", attack_methods)

    for attack_name in attack_methods:
        if attack_name == "word-d":
            attack = WordDeletion(ratio=0.2)
        elif attack_name == "typo":
            attack = TypoAttack(typo_rate=0.5)
        elif attack_name == "mispelling":
            attack = MispellingAttack(error_rate=0.5)
        elif attack_name == "contraction":
            attack = ContractionAttack()
        elif attack_name == "expansion":
            attack = ExpansionAttack()
        elif attack_name == "modify":
            attack = ModifyAttack(modification_prob=0.3)
        elif attack_name == "word-s":
            attack = SynonymSubstitution(ratio=0.8)
        elif attack_name == "word-s(Context)":
            attack = ContextAwareSynonymSubstitution(ratio=0.5,
                                                     tokenizer=BertTokenizer.from_pretrained('/model/bert-large-uncased'),
                                                     model=BertForMaskedLM.from_pretrained('/model/bert-large-uncased').to(device))
        elif attack_name == "doc-p(GPT-3.5)":
            attack = GPTParaphraser(openai_model="gpt-3.5-turbo",
                                    prompt="Please rewrite the following text: ",
                                    api_key="sk-...",#api_key and url(fill in NONE if not needed)
                                    api_base="api_base"
                                    )
        # elif attack_name.startswith("doc-p(") and attack_name.endswith(")"):

        #     model_name = attack_name[len("doc-p("):-1].strip()
        #     allowed_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo","deepseek-chat"]
        #     if model_name not in allowed_models:
        #     raise ValueError(f"Unsupported model: {model_name}")
        #     attack = GPTParaphraser( openai_model=model_name,
        #                             prompt="Please rewrite the following text: ",
        #                             api_key="sk-...",#api_key and url(fill in NONE if not needed)
        #                             api_base="api_base"
        #                             )

        elif attack_name == "doc-p(Dipper)":
            attack = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained("model/google/t5-v1_1-xxl/"),
                                       model=T5ForConditionalGeneration.from_pretrained("/dipper-paraphraser-xxl/", device_map="auto"),
                                       lex_diversity=60,
                                       order_diversity=0,
                                       sent_interval=1,
                                       max_new_tokens=100,
                                       do_sample=True,
                                       top_p=0.75,
                                       top_k=None)
        elif attack_name == "translation":
            attack = BackTranslationTextEditor(translate_to_intermediary=Translator(from_lang="en", to_lang="zh").translate,
                                               translate_to_source=Translator(from_lang="zh", to_lang="en").translate)
        else:
            print(f"Unknown attack method: {attack_name}")
            continue
        
    if attack_methods:
        attack_pipline = WatermarkedTextDetectionPipeline(dataset=my_dataset,
                                                          text_editor_list=[TruncatePromptTextEditor(), attack],
                                                          show_progress=True, return_type=DetectionPipelineReturnType.SCORES)
    else:
        attack_pipline = WatermarkedTextDetectionPipeline(dataset=my_dataset,
                                                          text_editor_list=[TruncatePromptTextEditor()],
                                                          show_progress=True,
                                                          return_type=DetectionPipelineReturnType.SCORES)
    default_pipline = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                         show_progress=True,
                                                         return_type=DetectionPipelineReturnType.SCORES)
    calculator = DynamicThresholdSuccessRateCalculator(labels=labels, rule=rules, target_fpr=target_fpr)
    print(calculator.calculate(attack_pipline.evaluate(myWatermark), default_pipline.evaluate(myWatermark)))


def test_evaluate_algorithm(algorithm_name, model_path, metric):
    
    # Model and tokenizer setting
    config = AutoConfig.from_pretrained(model_path)
    archs = getattr(config, "architectures", [])
    if not archs:
        raise ValueError("Unable to identify model type from config.json, please check if the model path is correct.")

    arch = archs[0].lower()
    is_seq2seq = any(k in arch for k in ("t5", "bart", "marian", "mbart", "pegasus", "mt5", "m2m100", "nllb", "switch", "byt5"))

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="deu_Latn")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # If the size of the vocabulary is inconsistent, adjust the model's vocab_size.
    if model.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
    else:
        pass

    # evluation
    if metric == 'PPL':
        my_dataset = C4Dataset('../data/dataset/c4/c4.json')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 vocab_size=tokenizer.vocab_size,
                                                 device=device,
                                                 max_new_tokens=200,
                                                 min_length=230,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)
        quality_pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset,
                                                             watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                             unwatermarked_text_editor_list=[],
                                                             analyzer=PPLCalculator(model=model,
                                                                                    tokenizer=tokenizer,
                                                                                    device=device),
                                                             unwatermarked_text_source='natural', show_progress=True,
                                                             return_type=QualityPipelineReturnType.MEAN_SCORES)
       

    elif metric == 'Log_Diversity':
        my_dataset = C4Dataset('../data/dataset/c4/c4.json')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 vocab_size=tokenizer.vocab_size,
                                                 device=device,
                                                 max_new_tokens=200,
                                                 min_length=230,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)
        quality_pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset,
                                                             watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                             unwatermarked_text_editor_list=[],
                                                             analyzer=LogDiversityAnalyzer(),
                                                             unwatermarked_text_source='natural', show_progress=True,
                                                             return_type=QualityPipelineReturnType.MEAN_SCORES)


    elif metric == 'BLEU':
        """Evaluate the impact on text quality in the machine translation task."""
        my_dataset = WMT16DE_ENDataset('../data/dataset/wmt16_de_en/wmt16_de_en.jsonl')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=tokenizer.vocab_size,
                                                 forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                 watermarked_text_editor_list=[],
                                                                 unwatermarked_text_editor_list=[],
                                                                 analyzer=BLEUCalculator(),
                                                                 unwatermarked_text_source='generated', show_progress=True,
                                                                 return_type=QualityPipelineReturnType.MEAN_SCORES)
    
    elif metric == 'Rouge':
        """Evaluate the impact on text quality in the machine translation task."""
        my_dataset = WMT16DE_ENDataset('../data/dataset/wmt16_de_en/wmt16_de_en.jsonl')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=tokenizer.vocab_size,
                                                 forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                 watermarked_text_editor_list=[],
                                                                 unwatermarked_text_editor_list=[],
                                                                 analyzer=RougeCalculator(),
                                                                 unwatermarked_text_source='generated', show_progress=True,
                                                                 return_type=QualityPipelineReturnType.MEAN_SCORES)

    elif metric == 'Mauve':
        """Evaluate the impact on text quality in the machine translation task."""
        my_dataset = WMT16DE_ENDataset('../data/dataset/wmt16_de_en/wmt16_de_en.jsonl')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=tokenizer.vocab_size,
                                                 forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                 watermarked_text_editor_list=[],
                                                                 unwatermarked_text_editor_list=[],
                                                                 analyzer=MauveCalculator(),
                                                                 unwatermarked_text_source='generated', show_progress=True,
                                                                 return_type=QualityPipelineReturnType.MEAN_SCORES)                                                            
    
    elif metric == 'BERTScore':  
        """Evaluate the impact on text quality in the machine translation task and text generation task.（最后返回F1）"""
        my_dataset = WMT16DE_ENDataset('../data/dataset/wmt16_de_en/wmt16_de_en.jsonl')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=tokenizer.vocab_size,
                                                 forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                 watermarked_text_editor_list=[],
                                                                 unwatermarked_text_editor_list=[],
                                                                 analyzer=BERTScoreCalculator(model_path="google-bert/bert-base-uncased"),  # If it is a Chinese text generation task, the rating model can be changed to "google-bert/bert-base-chinese".
                                                                 unwatermarked_text_source='generated', show_progress=True,
                                                                 return_type=QualityPipelineReturnType.MEAN_SCORES)
    
    elif metric == 'Pass@1':
        """Evaluate the impact on text quality in the code generation task."""
        my_dataset = HumanEvalDataset('../data/dataset/humaneval/humaneval.jsonl')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 min_length=200,
                                                 max_new_tokens=200)

        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                 watermarked_text_editor_list=[TruncateTaskTextEditor(),CodeGenerationTextEditor()],
                                                                 unwatermarked_text_editor_list=[TruncateTaskTextEditor(), CodeGenerationTextEditor()],
                                                                 analyzer=PassOrNotJudger(),
                                                                 unwatermarked_text_source='generated', show_progress=True,
                                                                 return_type=QualityPipelineReturnType.MEAN_SCORES)

    elif metric == 'GPT4_Judge':
        """Need to configure openai.api-key first."""
        my_dataset = WMT16DE_ENDataset('../data/dataset/wmt16_de_en/wmt16_de_en.jsonl')
        transformers_config = TransformersConfig(model=model,
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=tokenizer.vocab_size,
                                                 forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))

        quality_pipeline = ExternalDiscriminatorTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                            watermarked_text_editor_list=[],
                                                                            unwatermarked_text_editor_list=[],
                                                                            analyzer=GPTTextDiscriminator(openai_model='gpt-4',task_description='Translate the following German text to English'),
                                                                            unwatermarked_text_source='generated', show_progress=True,
                                                                            return_type=QualityPipelineReturnType.MEAN_SCORES)
    
    else:
        raise ValueError('Invalid metric')

    my_watermark = AutoWatermark.load(f'{algorithm_name}', algorithm_config=f'../config/watermark_config/{algorithm_name}.json', transformers_config=transformers_config)
    print(f"{metric}: ", quality_pipeline.evaluate(my_watermark))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default='none',
        help="The function of the different task: add_watermark/detect/evaluate.",
    )

    parser.add_argument(
        "--algorithm", 
        type=str, 
        default='KGW',
        help="The name of watermark algorithm.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="The path to the models. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--attack",
        type=str,
        nargs='*',
        default=[],
        help="The name of watermark attack algorithm.",
    )
  
    parser.add_argument(
        "--labels",
        nargs='+',
        default=['TPR', 'F1', 'ACC', 'AUROC']
    )
  
    parser.add_argument(
        "--rules",
        type=str,
        default='best'
    )
  
    parser.add_argument(
        "--target_fpr",
        type=float,
        default=0.01)
  
    parser.add_argument(
        "--metric",
        type=str,
        default='PPL',
        help="The metric of watermark evaluation.",
    )

    parser.add_argument(
        "--visualize_or_not",
        type=bool,
        default=False,
        help="Do you need to generate visual images?(True/False)",
    )

    args = parser.parse_args()
    if args.task == 'add_watermark':
        test_add_watermark_algorithm(args.algorithm, args.model_path, args.visualize_or_not)
    elif args.task == 'detect':  
        test_detect_algorithm(args.algorithm, args.model_path, args.attack, args.labels, args.rules, args.target_fpr)
    elif args.task  == 'evaluate':
        test_evaluate_algorithm(args.algorithm, args.model_path, args.metric)
