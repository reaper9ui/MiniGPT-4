from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat


#python minigpt.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
#conda activate minigpt4  
# Load config
cfg = Config(args)
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda')
vis_processor = registry.get_processor_class(model_config.vis_processor.name).from_config(model_config.vis_processor)

# Initialize chat
chat = Chat(model, vis_processor, device='cuda')

# Process image and get response
image_path = "path/to/your/image.jpg"
chat_state = chat.upload_img(image_path)
response = chat.answer(chat_state, "Describe what you see in this image.")
print(response)