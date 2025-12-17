from attack.privacy.attacks.DataExtraction.Conditional_DEA import Conditional_DEA
from attack.privacy.attacks.MIA.MIA import MIA
from attack.privacy.attacks.SD_MIA.SD_Attack import SD_Attack


def privacy_pipeline_demo(model=None, tokenizer=None, args=None):
    if args.attack == "C_DEA":
        Conditional_DEA(model, tokenizer, args)
    elif args.attack == "MIA":
        results = MIA(model, tokenizer, args)
        if getattr(args, 'metric', '').upper() == 'RECALL':
            print("ReCaLL metric completed with score summary:", results)
        return results
    elif args.attack == "SD_MIA":
        SD_Attack(args)
    else:
        raise ValueError(f"Invalid attack method: {args.attack}")
