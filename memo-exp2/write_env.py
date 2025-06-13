from verifiers import SingleTurnEnv
from write_parser import MEMOWriteXLMParser
from verifiers.rubrics import Rubric
from datasets import load_dataset


WRITE_SYSTEM_PROMPT = """You are a knowledge organizer assistant. Your task is to transform a passage into a structured folder-file system, \
summarizing or writing important content from the passage in a clear and concise manner to answer questions about the passage later. \
When given a long passage: 
1. Explain your step-by-step thought process and show your work and justify your approach inside <think>...</think> tags.
2. Folder-file structure must be:
<write>
<path>[folder name/file name]</path><content>[content]</content><path>[folder name/file name]</path><content>[content]</content>[...]
</write>

For example, if the passage is about mitochondria and cell membranes, you might output:
<think>
Mitochondria require two files: one for structure, one for function. Cell membrane needs one comprehensive file.
</think>
<write>
<path>Mitochondria/Structure.txt</path>\
<content>Double-membrane organelle with inner cristae folds. Contains mitochondrial DNA and ribosomes. Matrix contains metabolic enzymes...</content>\
<path>Cell_Membrane/Overview.txt</path>\
<content>
  Cell Membrane Overview  

  Structure:  
  - Phospholipid bilayer with hydrophilic heads facing outward and hydrophobic tails inward.  
  - Fluid mosaic model includes proteins and cholesterol embedded in the bilayer.  
  - Membrane proteins are either integral (spanning the membrane) or peripheral (attached to the surface).  

  Functions:  
  1. Selective permeability: Controls what enters/exits the cell (e.g., diffusion, osmosis, active transport).  
  2. Cell signaling: Receptor proteins detect external signals like hormones.  
  3. Compartmentalization: Separates the cell interior from its surroundings.  
  4. Cell adhesion: Proteins form junctions between cells (e.g., tight junctions, desmosomes).

  ...
</content>
</write>

Rules:
1. Folder names can repeat, but folder-file key pairs cannot repeat.
2. Do not include any additional text outside the <write> tag.
3. There should be only one <write> tag that includes all the <path> and <content> tags.
4. You cannot have nested <path> or <content> tags.
5. path and content tags must be in the same order, with each path tag followed by its corresponding content tag.
"""

class MEMOWriteEnv(SingleTurnEnv):
    
    def __init__(self, dataset_path='ehovy/race', max_words=1000):
        
        self.num_procs= 16
        dataset, eval_dataset = self.format_ds(dataset_path)
        parser = MEMOWriteXLMParser()
        rubric = Rubric(parser=parser)
        self.max_words = max_words
        self.memory = {}
        def memory_size_reward_func(completion, answer, **kwargs):
            memory_size = 0.0 
            for file in self.memory:
                memory_size += len(file)
            return 1 - memory_size / self.max_words if memory_size <= self.max_words else 0.0
        rubric.add_reward_func(memory_size_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=WRITE_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            message_type='chat'
            
        )
        
        
    def format_ds(self, dataset_path):
        def process(example):
            prompt = self.get_user_prompt_write(example['article'])
            return {'question': prompt}
        
        dataset = load_dataset(dataset_path, split='train', name='high')
        dataset = dataset.remove_columns(['question', 'options'])
        eval_dataset = load_dataset(dataset_path, split='validation', name='high')
        eval_dataset = eval_dataset.remove_columns(['options','question'])
        
        dataset = dataset.map(process, num_proc=self.num_procs)
        eval_dataset = eval_dataset.map(process, num_proc=self.num_procs)
        return dataset, eval_dataset
    
    
    def get_user_prompt_write(self, passage):
        return f"""Transform the following passage into a folder-file structure by analyzing the passage \
to identify key topics and subtopics, using think, write, path, and content tags.

Here is the passage:
{passage}

Good Luck!
"""
        
   
# testinggg!!!
from verifiers.inference.vllm_client import VLLMClient
client = VLLMClient(connection_timeout=800)    
vf_env = MEMOWriteEnv()
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

samplingparams = {
    "max_tokens": 4096,
    "repition_penalty": 1.05,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
}

def main():
    results = vf_env.evaluate(
        client=client,
        model=MODEL_ID,
        env=vf_env,
        sampling_params=samplingparams,

    )
    dataset_dsv3 = vf_env.make_dataset(results, extra_columns=['memory_size_reward_func', 'format_reward_func'])
        # save to hub
    dataset_dsv3.push_to_hub('Qwen2.5-7B-Instruct-memo-exp2-write-env')
        
if __name__ == "__main__":
    main()
        
