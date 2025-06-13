from verifiers.parsers import Parser, XMLParser

import re
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from types import SimpleNamespace
class MEMOWriteXLMParser(Parser):


    def __init__(self):
        super().__init__()
        self.write_main_parser = XMLParser(fields=["think", "write"])
        self.write_step_parser = XMLParser(fields=["path", "content"])
        
    def parse(self, text: str, strip:bool = True) -> Any:
       
        write_main_results = self.write_main_parser.parse(text, strip=strip)
        steps = []
        if write_main_results.write is None:
            write_main_results.write = ""
        for step_match in re.finditer(r'<path>(.*?)</path>\s*<content>(.*?)</content>', write_main_results.write, re.DOTALL):
            step_parsed = self.write_step_parser.parse(step_match.group(), strip=strip)
            steps.append(SimpleNamespace(path=step_parsed.path, content=step_parsed.content))
        return SimpleNamespace(
            think=write_main_results.think,
            write=steps,
        )


        
    def get_format_reward_func(self):
        
        def format_reward_func(completion):
            model_messages = self.get_assistant_messages(completion)
            write_main_reward = 0.0
            write_step_reward = 0.0

            try:
                write_main_reward += self.write_main_parser.get_format_reward_func()(model_messages)
                write_namespace = self.parse(model_messages[0]['content'], strip=True)
                num_tags_check = max(
                    len(re.findall(r'<path>', model_messages[0]['content'])),
                    len(re.findall(r'</path>', model_messages[0]['content'])),
                    len(re.findall(r'<content>', model_messages[0]['content'])),
                    len(re.findall(r'</content>', model_messages[0]['content'])))
                if len(write_namespace.write) > 0:
                    write_step_reward += self.write_step_parser.get_format_reward_func()(
                        [{'role': 'assistant', 'content': f"<path>{step.path}</path><content>{step.content}</content>"} 
                         for step in write_namespace.write]
                        )
                    write_step_reward *= (len(write_namespace.write) / num_tags_check) if num_tags_check > 0 else 1.0
                print(f"write main reward: {write_main_reward}, write step reward: {write_step_reward}")
            except Exception as e:
                pass 
            return 1/2 * (write_main_reward + write_step_reward)
           
  
        return format_reward_func    
    
            
            
                
parser = MEMOWriteXLMParser()
output = """<think>\nThe passage discusses the cultural significance of silence, focusing on various cultural perspectives and its implications in different scenarios. I will create folders for main cultural groups mentioned (Native Americans, Chinese, Thai, Russians, French, Spanish, Mexicans, Asians) and a general folder for nurses and patient care. Each folder will contain a file with the key points about that specific culture or group.\n</think>\n<write>\n <path>Cultural_Groups/Native_Americans/Overview.txt</path>\n <content>\n Native Americans value silence as a fundamental part of communication. Silence in conversation may imply a need for the listener to reflect on what has been said. It is seen as a call for deep thinking rather than an uncomfortable silence.\n </content>\n \n <path>Cultural_Groups/Chinese/Overview.txt</path>\n <content>\n Chinese culture views silence as a necessary part of communication. It is used to convey thoughtful pauses and reflect on the conversation. Silence is not seen as uncomfortable but rather as a moment for deeper understanding.\n </content>\n \n <path>Cultural_Groups/Thai/Overview.txt</path>\n <content>\n Thai culture also values silence as a key component of effective communication. It is seen as a way to show respect and thoughtful consideration rather than an uncomfortable pause.\n </content>\n \n <path>Cultural_Groups/Russians/Overview.txt</path>\n <content>\n Russians may use silence to signify agreement during discussions. It is a way to show consent without verbal affirmation.\n </content>\n \n <path>Cultural_Groups/French/Overview.txt</path>\n <content>\n French culture also employs silence to indicate agreement. It is a subtle way to communicate without explicitly stating one's position.\n </content>\n \n <path>Cultural_Groups/Spanish/Overview.txt</path>\n <content>\n Spanish culture uses silence similarly to sign agreement in discussions. It is a non-verbal method of expressing consent.\n </content>\n \n <path>Cultural_Groups/Mexicans/Overview.txt</path>\n <content>\n In Mexican culture, silence is a sign of respect when instructions are given by someone in authority. It is a way to show compliance without arguing.\n </content>\n \n <path>Cultural_Groups/Asians/Overview.txt</path>\n <content>\n Asian cultures often view silence as a sign of respect, particularly towards elders or those in authority. It is used to show reverence and thoughtful consideration.\n </content>\n \n <path/Nurses_and_Patient_Care/Overview.txt</path>\n <content>\n Nurses and caregivers must recognize the cultural significance of silence when dealing with patients. Understanding the potential meanings of silence can help in providing appropriate care and support. Nurses should be aware of their own personal and cultural views on silence to avoid misinterpreting their patients' silence.\n </content>\n</write>"""

print(parser.get_format_reward_func()([{'role': 'assistant', 'content': output}]))
content = parser.write_main_parser.parse(output, strip=True)

# print(f"think: {content.think}")
# print('\n')
# print(f"write: {content.write}")
print(parser.write_main_parser.get_format_str())