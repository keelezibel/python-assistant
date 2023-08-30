import time
import json
import requests
import websockets
import gradio as gr
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional

llm_host = "http://130.16.61.251:5001/api/v1/generate"
user_msg_token = "### Instruction"
asst_msg_token = "### Response"

class CustomLLM(LLM):
    temperature: float
    max_new_tokens: int
    seed: int
    system_prompt: str
    
    def text_wrapper(self, prompt):
        return self.system_prompt + f"\n{user_msg_token}: {prompt}\n{asst_msg_token}:"
    
    async def generate_webui(self, input_text: str):
        request = {
            'prompt': input_text,
            'max_new_tokens': self.max_new_tokens,
            'preset': 'None',
            'do_sample': True,
            'temperature': self.temperature,
            'repetition_penalty': 1.10,
            'num_beams': 1,
            'early_stopping': False,
            'seed': self.seed,
            'truncation_length': 4096
        }
        
        async with websockets.connect(llm_host, ping_interval=None) as ws:
            await ws.send(json.dumps(request))

            while True:
                incoming_data = await websocket.recv()
                incoming_data = json.loads(incoming_data)
                match incoming_data['event']:
                    case 'text_stream':
                        yield incoming_data['text']
                    case 'stream_end':
                        return

    async def _generator(self, text: str, temperature: float, max_new_tokens: int, seed: int):
        input_text = self.text_wrapper(text)
        async for response in self.generate_webui(input_text):
            printsys.stdout.flush()
        # yield self.generate_webui(input_text)
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    async def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
            #   run_manager: Optional[CallbackManagerForLLMRun] = None,
             ) -> str:
        if stop is not None:
            yield ValueError("stop kwargs are not permitted")
        
        yield self._generator(prompt, temperature=self.temperature, max_new_tokens=self.max_new_tokens, seed=self.seed)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "system_prompt": self.system_prompt
        }

class GradioInference():
    def __init__(self):
        system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        self.llm = CustomLLM(temperature=0.1, max_new_tokens=2048, seed=1000, system_prompt=system_prompt)

    def prep_prompt(self, instruction, code):
        return f"{instruction}\n\nCode:\n```python\n{code}\n```"

    def __call__(self, history, chat_state, instruction: str, code: str):
        if instruction == "Add Unit Tests":
            instruction_prompt = "Write unit tests for the following code."
        print(f"Chat State: {chat_state}")
        print(f"Instruction: {instruction}")
        print(f"Code: {code}")
        prompt = self.prep_prompt(instruction_prompt, code)
        # print(prompt)
        history[-1][-1] = ''
        for char in self.llm(prompt):
            # print(char)
            history[-1][-1] += char
            time.sleep(0.05)
            yield history
        # return self.llm(self.prep_prompt(instruction_prompt, code))

gio = GradioInference()
title = """
    <div style="text-align: center; max-width:500px; margin: 0 auto;">
        <div>
        <h1>Cody Python</h1>
        </div>
    </div>
"""

with gr.Blocks() as demo:
    gr.HTML(title)
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=0.5):
            instruction = gr.Dropdown(["Add Unit Tests"], label="Instructions")
            code = gr.Textbox(label="Your Code")
            # btn = gr.Button("Leggo")
        with gr.Column(scale=0.5):
            # output = gr.Textbox(lines=15, label="Output")
            chat_state = gr.State()
            chatbot = gr.Chatbot()
            clear = gr.Button("Clear")

    def user(user_message, history):
        return '', history + [[user_message, None]]
    
    code.submit(user, [code, chatbot], [code, chatbot], queue=False).then(
        gio, [chatbot, chat_state, instruction, code], [chatbot]
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # btn.click(gio, inputs=[instruction, code], outputs=[output])

# demo.queue(concurrency_count=3)
demo.queue()
demo.launch()