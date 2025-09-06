"""
ATC Scoring System for PyTorch
"""
import numpy as np
import cv2
from config import ATC_SCORING_CRITERIA

class ATCScorer:
    """Animal Type Classification Scoring System"""
    
    def __init__(self):
        self.scoring_criteria = ATC_SCORING_CRITERIA
        print("🐄 ATC Scorer initialized")
    
    def extract_body_parameters(self, image):
        """Extract body measurements from image"""
        try:
            # Convert tensor to numpy if needed
            if hasattr(image, 'cpu'):
                image = image.cpu().numpy()
            
            # Ensure image is in correct format
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Edge detection and contour finding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return self._get_default_parameters()
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate measurements
            img_height, img_width = image.shape[:2]
            body_length = (w / img_width) * 155 + np.random.normal(0, 3)
            height_at_withers = (h / img_height) * 130 + np.random.normal(0, 2)
            chest_width = (w * 0.35 / img_width) * 45 + np.random.normal(0, 1)
            
            # Functional assessments
            rump_angle = self._calculate_rump_angle(largest_contour)
            udder_score = self._assess_udder_quality(gray, x, y, w, h)
            leg_score = self._assess_leg_structure(gray, x, y, w, h)
            overall_score = (udder_score + leg_score) / 2
            
            # Constrain values
            parameters = {
                'body_length': max(120, min(180, round(float(body_length), 1))),
                'height_at_withers': max(110, min(150, round(float(height_at_withers), 1))),
                'chest_width': max(35, min(55, round(float(chest_width), 1))),
                'rump_angle': max(15, min(35, round(float(rump_angle), 1))),
                'udder_attachment': max(1, min(9, int(udder_score))),
                'leg_structure': max(1, min(9, int(leg_score))),
                'overall_conformation': max(1, min(9, round(float(overall_score), 1)))
            }
            
            return parameters
            
        except Exception as e:
            print(f"Error in body parameter extraction: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self):
        """Default parameters when extraction fails"""
        return {
            'body_length': 150.0,
            'height_at_withers': 125.0,
            'chest_width': 45.0,
            'rump_angle': 25.0,
            'udder_attachment': 5,
            'leg_structure': 5,
            'overall_conformation': 5.0
        }
    
    def _calculate_rump_angle(self, contour):
        """Calculate rump angle"""
        try:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                return 15 + (ellipse[2] % 90) * 20 / 90
        except:
            pass
        return 25.0
    
    def _assess_udder_quality(self, gray, x, y, w, h):
        """Assess udder quality"""
        try:
            lower_roi = gray[y + int(h*0.7):y + h, x:x + w]
            if lower_roi.size > 0:
                texture_var = cv2.Laplacian(lower_roi, cv2.CV_64F).var()
                return max(1, min(9, int(3 + texture_var / 200)))
        except:
            pass
        return 5
    
    def _assess_leg_structure(self, gray, x, y, w, h):
        """Assess leg structure"""
        try:
            leg_roi = gray[y + int(h*0.6):y + h, x:x + w]
            if leg_roi.size > 0:
                edges = cv2.Canny(leg_roi, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20)
                line_count = len(lines) if lines is not None else 0
                return max(1, min(9, int(3 + line_count / 5)))
        except:
            pass
        return 5
    
    def calculate_atc_score(self, parameters):
        """Calculate ATC score"""
        total_score = 0
        detailed_scores = {}
        
        for param, value in parameters.items():
            if param in self.scoring_criteria:
                criteria = self.scoring_criteria[param]
                
                if param in ['udder_attachment', 'leg_structure', 'overall_conformation']:
                    normalized = (float(value) - 1) / 8 * 100
                else:
                    normalized = (float(value) - criteria['min']) / (criteria['max'] - criteria['min']) * 100
                    normalized = max(0, min(100, normalized))
                
                weighted_score = normalized * criteria['weight']
                total_score += weighted_score
                
                detailed_scores[param] = {
                    'value': float(value),
                    'unit': criteria['unit'],
                    'normalized': round(float(normalized), 1),
                    'weighted': round(float(weighted_score), 2)
                }
        
        return {
            'total_score': round(float(total_score), 2),
            'grade': self._assign_grade(total_score),
            'detailed_scores': detailed_scores
        }
    
    def _assign_grade(self, score):
        """Assign grade"""
        if score >= 90: return 'A+ (Excellent)'
        elif score >= 80: return 'A (Very Good)'
        elif score >= 70: return 'B (Good)'
        elif score >= 60: return 'C (Average)'
        else: return 'D (Below Average)'
