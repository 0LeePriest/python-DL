import gradio as gr
# def increase(num):
#     return num + 1
# with gr.Blocks() as demo:
#     a = gr.Number(label="a")
#     b = gr.Number(label="b")
#     # 要想b>a，则使得b = a+1
#     atob = gr.Button("b > a")
#     atob.click(increase, a, b)
#     # 要想a>b，则使得a = b+1
#     btoa = gr.Button("a > b")
#     btoa.click(increase, b, a)
# demo.launch()
# import fire
#
#
#
#
# def process_list(items):
#     return [item.upper() for item in items]
#
#
# if __name__ == '__main__':
#     fire.Fire(process_list)
import sys
import torch
from transformers import GenerationConfig, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
# def main():

    # Loda data
    # all_data_samples = []
    # datasets = ['SAMSum']
    # for dataset in datasets:
    #     data_in_dataset_class = load_dataset('binwang/InstructDS_datasets', dataset, split='validation')
    #     print(data_in_dataset_class)
    #     # print("**************************")
    #     # for sample in data_in_dataset_class:
    #     #     print(sample)
    #     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     data = [sample for sample in data_in_dataset_class]
    #     print(len(data))
def main():
        model_path = 'binwang/InstructDS'



        # Load model
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        temperature = 0.7
        top_p = 0.75
        top_k = 40
        num_beams = 8
        max_new_tokens = 128
        instruction = 'Please summarize the following dialogue from SAMSum dataset.'
        dialogue ="Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nHannah: <file_gif>\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye"

        input_to_model = '###Instruction:\n{}\n### Input:\n{}\n'.format(instruction, dialogue)



        print(input_to_model)

        inputs = tokenizer(input_to_model, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda')

        generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=True,

            )

        with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
        output_sequence = generation_output.sequences[0]
        output = tokenizer.decode(output_sequence, skip_special_tokens=True)
        print(output)

main()
