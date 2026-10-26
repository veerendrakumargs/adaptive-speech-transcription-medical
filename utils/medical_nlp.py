"""
Medical NLP utilities using SciSpacy for biomedical entity recognition
Extracts diseases, medications, procedures, and anatomy from medical transcripts
"""
import spacy
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MedicalNLPProcessor:
    """Advanced biomedical NLP for clinical term extraction"""
    
    ENTITY_TYPES = {
        'DISEASE': 'Diseases and Conditions',
        'CHEMICAL': 'Medications and Chemicals',
        'GENE': 'Genes and Proteins',
        'ANATOMY': 'Anatomical Structures',
        'PROCEDURE': 'Medical Procedures',
        'SYMPTOM': 'Symptoms'
    }
    
    def __init__(self, model_name='en_core_web_sm'):
        """
        Initialize SciSpacy NLP model
        Note: For production, use en_core_sci_sm or en_ner_bc5cdr_md
        """
        self.model_name = model_name
        self.nlp = None
        self.load_model()
    
    def load_model(self):
        """Load SciSpacy model"""
        try:
            # Try to load scispacy model first
            try:
                self.nlp = spacy.load('en_core_sci_sm')
                self.model_name = 'en_core_sci_sm'
            except:
                # Fallback to standard spacy model
                try:
                    self.nlp = spacy.load('en_core_web_sm')
                    self.model_name = 'en_core_web_sm'
                except:
                    # Download and load
                    import subprocess
                    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
                    self.nlp = spacy.load('en_core_web_sm')
                    self.model_name = 'en_core_web_sm'
            
            print(f"Loaded NLP model: {self.model_name}")
        except Exception as e:
            raise Exception(f"Failed to load NLP model: {e}")
    
    def extract_entities(self, text):
        """
        Extract biomedical entities from text
        Returns: List of entities with type, text, and position
        """
        if self.nlp is None:
            self.load_model()
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': self.ENTITY_TYPES.get(ent.label_, ent.label_)
            })
        
        return entities
    
    def highlight_clinical_terms(self, text, entities=None):
        """
        Add HTML highlighting to clinical terms
        Args:
            text: Original text
            entities: Pre-extracted entities (optional)
        Returns:
            HTML formatted text with highlighted terms
        """
        if entities is None:
            entities = self.extract_entities(text)
        
        # Sort entities by start position in reverse to avoid position shifts
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        highlighted_text = text
        color_map = {
            'DISEASE': '#ff6b6b',
            'CHEMICAL': '#4ecdc4',
            'ANATOMY': '#45b7d1',
            'PROCEDURE': '#f7b731',
            'GENE': '#5f27cd',
            'SYMPTOM': '#ff9ff3',
            'PERSON': '#48dbfb',
            'ORG': '#54a0ff',
            'GPE': '#00d2d3'
        }
        
        for entity in sorted_entities:
            color = color_map.get(entity['label'], '#95afc0')
            replacement = f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white;">{entity["text"]}</mark>'
            highlighted_text = (
                highlighted_text[:entity['start']] + 
                replacement + 
                highlighted_text[entity['end']:]
            )
        
        return highlighted_text
    
    def categorize_entities(self, entities):
        """Categorize entities by type"""
        categorized = {}
        for entity in entities:
            label = entity['label']
            if label not in categorized:
                categorized[label] = []
            categorized[label].append(entity['text'])
        return categorized
    
    def extract_medical_concepts(self, text):
        """
        Extract key medical concepts including:
        - Diseases/Conditions
        - Medications
        - Procedures
        - Symptoms
        - Anatomy
        """
        entities = self.extract_entities(text)
        categorized = self.categorize_entities(entities)
        
        concepts = {
            'diseases': categorized.get('DISEASE', []),
            'medications': categorized.get('CHEMICAL', []),
            'procedures': categorized.get('PROCEDURE', []),
            'symptoms': categorized.get('SYMPTOM', []),
            'anatomy': categorized.get('ANATOMY', []),
            'all_entities': entities
        }
        
        return concepts
    
    def get_entity_statistics(self, text):
        """Get statistics about extracted entities"""
        entities = self.extract_entities(text)
        categorized = self.categorize_entities(entities)
        
        stats = {
            'total_entities': len(entities),
            'unique_types': len(categorized),
            'by_type': {k: len(v) for k, v in categorized.items()}
        }
        
        return stats
    
    def process_transcript(self, transcript_text, highlight=True):
        """
        Process complete medical transcript
        Returns: entities, highlighted text, statistics
        """
        entities = self.extract_entities(transcript_text)
        concepts = self.extract_medical_concepts(transcript_text)
        stats = self.get_entity_statistics(transcript_text)
        
        result = {
            'entities': entities,
            'concepts': concepts,
            'statistics': stats
        }
        
        if highlight:
            result['highlighted_text'] = self.highlight_clinical_terms(transcript_text, entities)
        
        return result
