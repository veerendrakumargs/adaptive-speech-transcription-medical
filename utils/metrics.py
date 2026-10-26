"""
Evaluation metrics for speech recognition
Includes WER, CER, accuracy, precision, recall, F1-score
Based on research in end-to-end speech recognition (Chiu et al., 2018)
"""
import jiwer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import List, Dict, Tuple

class TranscriptionMetrics:
    """Comprehensive evaluation metrics for speech recognition"""
    
    def __init__(self):
        self.transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords()
        ])
    
    def calculate_wer(self, reference, hypothesis):
        """
        Calculate Word Error Rate (WER)
        WER = (S + D + I) / N
        where S=substitutions, D=deletions, I=insertions, N=total words
        """
        try:
            wer = jiwer.wer(reference, hypothesis, truth_transform=self.transformation,
                           hypothesis_transform=self.transformation)
            return wer
        except:
            return 1.0
    
    def calculate_cer(self, reference, hypothesis):
        """Calculate Character Error Rate (CER)"""
        try:
            cer = jiwer.cer(reference, hypothesis)
            return cer
        except:
            return 1.0
    
    def calculate_word_metrics(self, reference, hypothesis):
        """
        Calculate detailed word-level metrics
        Returns: substitutions, deletions, insertions, hits
        """
        try:
            output = jiwer.process_words(
                reference, hypothesis,
                truth_transform=self.transformation,
                hypothesis_transform=self.transformation
            )
            return {
                'substitutions': output.substitutions,
                'deletions': output.deletions,
                'insertions': output.insertions,
                'hits': output.hits,
                'wer': output.wer,
                'mer': output.mer,  # Match Error Rate
                'wil': output.wil   # Word Information Lost
            }
        except:
            return {
                'substitutions': 0,
                'deletions': 0,
                'insertions': 0,
                'hits': 0,
                'wer': 1.0,
                'mer': 1.0,
                'wil': 1.0
            }
    
    def calculate_accuracy(self, reference, hypothesis):
        """Calculate transcription accuracy (1 - WER)"""
        wer = self.calculate_wer(reference, hypothesis)
        return max(0, 1 - wer)
    
    def calculate_sentence_accuracy(self, references, hypotheses):
        """
        Calculate sentence-level accuracy
        Percentage of perfectly matched sentences
        """
        if not references or not hypotheses:
            return 0.0
        
        correct = sum(1 for ref, hyp in zip(references, hypotheses) 
                     if ref.strip().lower() == hyp.strip().lower())
        return correct / len(references)
    
    def calculate_bleu_score(self, reference, hypothesis):
        """Calculate BLEU score for translation quality"""
        from nltk.translate.bleu_score import sentence_bleu
        try:
            reference_tokens = reference.lower().split()
            hypothesis_tokens = hypothesis.lower().split()
            score = sentence_bleu([reference_tokens], hypothesis_tokens)
            return score
        except:
            return 0.0
    
    def calculate_comprehensive_metrics(self, reference, hypothesis):
        """
        Calculate all metrics for a reference-hypothesis pair
        Returns comprehensive dictionary of metrics
        """
        word_metrics = self.calculate_word_metrics(reference, hypothesis)
        
        metrics = {
            'wer': self.calculate_wer(reference, hypothesis),
            'cer': self.calculate_cer(reference, hypothesis),
            'accuracy': self.calculate_accuracy(reference, hypothesis),
            'word_details': word_metrics,
            'reference_length': len(reference.split()),
            'hypothesis_length': len(hypothesis.split())
        }
        
        # Calculate precision, recall, F1 at word level
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        
        if hyp_words:
            metrics['precision'] = len(ref_words & hyp_words) / len(hyp_words)
        else:
            metrics['precision'] = 0.0
        
        if ref_words:
            metrics['recall'] = len(ref_words & hyp_words) / len(ref_words)
        else:
            metrics['recall'] = 0.0
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                 (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        return metrics
    
    def batch_evaluate(self, reference_list, hypothesis_list):
        """
        Evaluate multiple reference-hypothesis pairs
        Returns: individual metrics and aggregate statistics
        """
        if len(reference_list) != len(hypothesis_list):
            raise ValueError("Reference and hypothesis lists must have same length")
        
        all_metrics = []
        for ref, hyp in zip(reference_list, hypothesis_list):
            metrics = self.calculate_comprehensive_metrics(ref, hyp)
            all_metrics.append(metrics)
        
        # Calculate aggregate metrics
        aggregate = {
            'mean_wer': np.mean([m['wer'] for m in all_metrics]),
            'mean_cer': np.mean([m['cer'] for m in all_metrics]),
            'mean_accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'mean_precision': np.mean([m['precision'] for m in all_metrics]),
            'mean_recall': np.mean([m['recall'] for m in all_metrics]),
            'mean_f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'std_wer': np.std([m['wer'] for m in all_metrics]),
            'sentence_accuracy': self.calculate_sentence_accuracy(reference_list, hypothesis_list),
            'total_samples': len(all_metrics)
        }
        
        return {
            'individual_metrics': all_metrics,
            'aggregate_metrics': aggregate
        }
    
    def generate_confusion_matrix_data(self, references, hypotheses):
        """Generate data for confusion matrix visualization"""
        # Simplified confusion matrix for speech recognition
        # Categories: Perfect, Good (WER < 0.2), Fair (WER < 0.5), Poor (WER >= 0.5)
        categories = []
        for ref, hyp in zip(references, hypotheses):
            wer = self.calculate_wer(ref, hyp)
            if wer == 0:
                categories.append('Perfect')
            elif wer < 0.2:
                categories.append('Good')
            elif wer < 0.5:
                categories.append('Fair')
            else:
                categories.append('Poor')
        
        return categories
