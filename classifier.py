"""
Complete PyTorch ATC Classification System
"""
import torch
import numpy as np
from torchvision import transforms
from config import DEVICE
import json
import os
from datetime import datetime

class CattleATCSystem:
    """Complete Cattle Classification and ATC System"""
    
    def __init__(self, model, label_encoder, atc_scorer):
        self.model = model
        self.label_encoder = label_encoder
        self.atc_scorer = atc_scorer
        self.results_history = []
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("🐄 Complete ATC System initialized")
    
    def predict_breed(self, image, top_k=3):
        """Predict breed from image"""
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess image
            if len(image.shape) == 3:
                if image.shape[2] == 3:  # RGB format
                    input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
                else:  # Already tensor format
                    input_tensor = image.unsqueeze(0).to(DEVICE)
            else:
                input_tensor = image.to(DEVICE)
            
            # Get predictions
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_k):
                breed_idx = top_indices[0][i].cpu().item()
                confidence = top_probs[0][i].cpu().item()
                breed_name = self.label_encoder.inverse_transform([breed_idx])[0]
                results.append({'breed': breed_name, 'confidence': confidence})
            
            return results
    
    def classify_animal(self, image, animal_id=None):
        """Complete classification pipeline"""
        if animal_id is None:
            animal_id = f"ANIMAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Breed prediction
            breed_predictions = self.predict_breed(image, top_k=3)
            primary_breed = breed_predictions[0]['breed']
            primary_confidence = breed_predictions[0]['confidence']
            
            # Extract body parameters
            body_params = self.atc_scorer.extract_body_parameters(image)
            
            # Calculate ATC score
            atc_score = self.atc_scorer.calculate_atc_score(body_params)
            
            # Create result
            result = {
                'animal_id': animal_id,
                'timestamp': datetime.now().isoformat(),
                'breed_prediction': primary_breed,
                'confidence': primary_confidence,
                'top_predictions': breed_predictions,
                'body_parameters': body_params,
                'atc_score': atc_score
            }
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return None
    
    def save_results(self, filename='pytorch_atc_results.json'):
        """Save results to JSON"""
        os.makedirs('outputs', exist_ok=True)
        filepath = os.path.join('outputs', filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results_history, f, indent=2, default=str)
            print(f"📊 Results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def generate_report(self, result):
        """Generate ATC report"""
        if not result:
            return "Error generating report"
        
        report = f"""
{'='*80}
                    AI-BASED ANIMAL TYPE CLASSIFICATION REPORT
                          Rashtriya Gokul Mission (RGM) - PyTorch Edition
{'='*80}

ANIMAL IDENTIFICATION:
Animal ID: {result['animal_id']}
Classification Date: {result['timestamp']}

BREED CLASSIFICATION:
Primary Breed: {result['breed_prediction']} ({result['confidence']:.2%})

Top 3 Predictions:"""
        
        for i, pred in enumerate(result['top_predictions'], 1):
            report += f"\n  {i}. {pred['breed']}: {pred['confidence']:.2%}"
        
        report += f"""

BODY STRUCTURE MEASUREMENTS:
{'-'*50}"""
        
        for param, value in result['body_parameters'].items():
            unit = self.atc_scorer.scoring_criteria.get(param, {}).get('unit', '')
            report += f"\n{param.replace('_', ' ').title()}: {value} {unit}"
        
        report += f"""

ATC SCORING SUMMARY:
{'-'*50}
Total ATC Score: {result['atc_score']['total_score']}/100
Grade: {result['atc_score']['grade']}

System: PyTorch GPU-Accelerated ATC v2.0
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
        """
        
        return report
