import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipModel, BertModel, BertTokenizer

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device

        # Visual feature extractor (ResNet-101)
        resnet = models.resnet101(pretrained=True)
        self.visual_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.visual_extractor.eval()

        # BLIP for caption generation
        self.blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        self.blip_model = BlipModel.from_pretrained('Salesforce/blip-image-captioning-base').to(device)
        self.blip_model.eval()

        # BERT for text embedding
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert_model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_visual_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.visual_extractor(img_tensor).squeeze().cpu()
        return feat

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
        return caption

    def extract_textual_features(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

    def extract_pair_centric_features(self, personA_img, personB_img):
        featA = self.extract_visual_features(personA_img)
        featB = self.extract_visual_features(personB_img)
        return torch.cat([featA, featB], dim=0)

    def extract_scene_centric_features(self, scene_img):
        visual_feat = self.extract_visual_features(scene_img)
        caption = self.generate_caption(scene_img)
        text_feat = self.extract_textual_features(caption)
        return visual_feat, text_feat
