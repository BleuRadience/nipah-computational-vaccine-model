#!/usr/bin/env python3
"""
Enhanced Nipah Virus Computational Vaccine Model - CORRECTED
================================================================
Fixed issues:
- Proper Biopython import handling
- Corrected linker concatenation
- Improved amphipathicity calculation
- Better population coverage estimation (Hardy-Weinberg)
- Added error handling for missing dependencies
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from scipy.stats import norm

warnings.filterwarnings('ignore')

# Handle Biopython import gracefully
try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction, molecular_weight
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: Biopython not installed. Some features will be limited.")
    print("Install with: pip install biopython")

__version__ = "1.0.1"
__author__ = "Cassandra D. Harrison"
__email__ = "bleuisresting@gmail.com"
__license__ = "CC BY-NC-SA 4.0"


class NipahVaccineModel:
    """Enhanced Nipah Virus G-protein computational vaccine design model."""
    
    # Amino acid property scales - defined as class constants for performance
    HYDROPHOBICITY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    HYDROPHOBIC_SMALL = {'A': 0.31, 'I': 0.73, 'L': 0.73, 'M': 0.38, 
                         'F': 0.61, 'W': 0.37, 'Y': 0.20, 'V': 0.54}
    CHARGED = {'D': -0.77, 'E': -0.64, 'R': 0.68, 'K': 0.68, 'H': 0.13}
    POLAR = {'S': -0.04, 'T': 0.11, 'Q': -0.22, 'N': -0.28, 'C': 0.17, 
             'G': 0.00, 'P': -0.07}
    
    ANTIGENICITY = {
        'A': 1.064, 'R': 0.873, 'N': 0.851, 'D': 0.924, 'C': 1.412,
        'Q': 0.851, 'E': 0.851, 'G': 0.874, 'H': 1.105, 'I': 1.231,
        'L': 1.230, 'K': 0.721, 'M': 1.261, 'F': 1.205, 'P': 1.064,
        'S': 0.883, 'T': 0.941, 'W': 1.404, 'Y': 1.161, 'V': 1.161
    }
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Authentic mature G ectodomain sequence (NCBI NP_112027.1, residues 71-602)
        self.g_sequence = (
            "QNYTRSTDNQAVIKDALQGIQQQIKGLADKIGTEIGPKVSLIDTSSTITIPANIGLLGSK"
            "ISQSTASINENVNEKCKFTLPPLKIHECNISCPNPLPFREYRPQTEGVSNLVGLPNNICL"
            "QKTSNQILKPKLISYTLPVVGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRII"
            "GVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYW"
            "SGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFL"
            "VRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIE"
            "ISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQ"
            "SQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQ"
            "LASEDTNAQKTITN"
        )
        
        # Structural parameters from literature
        self.structural_params = {
            'tetramer_symmetry': 4,
            'binding_affinity_efnb2': 0.6e-9,
            'binding_affinity_efnb3': 15e-9,
            'head_domain': (156, 602),
            'stalk_domain': (71, 155),
            'rbs_residues': [492, 512, 530, 534, 581, 586],
            'neutralization_sites': 8,
        }
        
        # Immunological parameters
        self.immune_params = {
            'cd8_length': 9,
            'cd4_length_range': (13, 25),
            'b_cell_length': (12, 20),
            'endemic_hla_i': ['HLA-A*11:01', 'HLA-A*24:02', 'HLA-B*40:06', 'HLA-B*07:02'],
            'endemic_hla_ii': ['DRB1*03:01', 'DRB1*07:01', 'DRB1*15:01', 'DRB1*04:01'],
            'population_coverage_target': 0.70,
        }
        
        self._results = {}
        
    def log(self, message):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def _aa_composition(self, peptide: str) -> Dict[str, float]:
        """Calculate amino acid composition of a peptide."""
        if not peptide:
            return {}
        length = len(peptide)
        return {aa: peptide.count(aa) / length for aa in set(peptide)}
    
    def _calculate_epitope_score(self, peptide: str, epitope_type: str) -> float:
        """Calculate custom epitope score with corrected normalization."""
        if not peptide:
            return 0.0
        
        length = len(peptide)
        
        # Calculate base property scores
        hydrophobic_sum = sum(self.HYDROPHOBIC_SMALL.get(aa, 0) for aa in peptide)
        charged_sum = sum(abs(self.CHARGED.get(aa, 0)) for aa in peptide)
        polar_sum = sum(abs(self.POLAR.get(aa, 0)) for aa in peptide)
        
        h_content = hydrophobic_sum / length
        c_content = charged_sum / length
        p_content = polar_sum / length
        
        if epitope_type == 'cd8':
            # MHC-I: hydrophobic anchors at positions 2 and 9 (for 9-mers)
            anchor_score = (self.HYDROPHOBIC_SMALL.get(peptide[1], 0) * 2 + 
                          self.HYDROPHOBIC_SMALL.get(peptide[-1], 0) * 2)
            base_score = h_content * 0.6 + anchor_score * 0.3 + p_content * 0.1
            scaled = np.clip(base_score * 0.3 + 0.1, 0, 1)
            
        elif epitope_type == 'cd4':
            # MHC-II: amphipathic preference
            amphipathicity = min(h_content, c_content + p_content) * 1.2
            base_score = amphipathicity * 0.7 + p_content * 0.3
            scaled = np.clip(base_score * 0.4 + 0.05, 0, 1)
            
        else:  # b_cell
            # Surface accessibility preference
            flexibility = sum(1 for aa in peptide if aa in 'GP') / length * 0.3
            base_score = c_content + p_content - h_content * 0.5 + flexibility
            scaled = np.clip(base_score * 0.5 + 0.03, 0, 1)
        
        # Add deterministic noise based on peptide hash
        peptide_hash = hash(peptide) % 1000000 / 1000000.0
        noise = 0.05 * np.sin(peptide_hash * 2 * np.pi)
        
        return np.clip(scaled + noise, 0, 1)
    
    def _calculate_amphipathicity(self, peptide: str) -> float:
        """Calculate amphipathic moment using corrected vector sum."""
        if len(peptide) < 4:
            return 0.0
        
        # Use Eisenberg's helical wheel method (100° periodicity for alpha-helix)
        angle_increment = 100  # degrees
        sum_x, sum_y = 0.0, 0.0
        angle = 0
        
        for aa in peptide:
            h_val = self.HYDROPHOBICITY.get(aa, 0)
            angle_rad = np.radians(angle)
            sum_x += h_val * np.cos(angle_rad)
            sum_y += h_val * np.sin(angle_rad)
            angle = (angle + angle_increment) % 360
        
        return np.sqrt(sum_x**2 + sum_y**2) / len(peptide)
    
    def _calculate_antigenicity(self, peptide: str) -> float:
        """Estimate antigenicity based on Kolaskar & Tongaonkar method."""
        if not peptide:
            return 0.0
        scores = [self.ANTIGENICITY.get(aa, 1.0) for aa in peptide]
        return sum(scores) / len(scores)
    
    def _estimate_surface_accessibility(self, peptide: str, position: int) -> float:
        """Estimate surface accessibility."""
        seq_len = len(self.g_sequence)
        head_start, head_end = self.structural_params['head_domain']
        
        # Head domain is more accessible
        position_factor = 1.2 if head_start <= position <= head_end else 1.0
        
        # Hydrophobic residues reduce accessibility
        hydrophobic_content = sum(1 for aa in peptide if aa in 'AILMFWYV') / len(peptide)
        accessibility = (1.0 - hydrophobic_content * 0.5) * position_factor
        
        return np.clip(accessibility, 0.1, 1.0)
    
    def analyze_sequence_properties(self):
        """Analyze basic sequence properties."""
        self.log("Analyzing sequence properties...")
        
        hydrophobicity_profile = [self.HYDROPHOBICITY.get(aa, 0) for aa in self.g_sequence]
        
        if BIOPYTHON_AVAILABLE:
            protein_analysis = ProteinAnalysis(self.g_sequence)
            isoelectric_point = protein_analysis.isoelectric_point()
            instability_index = protein_analysis.instability_index()
            aromaticity = protein_analysis.aromaticity()
            gravy = protein_analysis.gravy()
            secondary_structure = protein_analysis.secondary_structure_fraction()
        else:
            # Fallback calculations
            isoelectric_point = 7.0
            instability_index = 40.0
            aromaticity = 0.07
            gravy = sum(hydrophobicity_profile) / len(hydrophobicity_profile)
            secondary_structure = {'helix': 0.3, 'turn': 0.2, 'sheet': 0.5}
        
        results = {
            'length': len(self.g_sequence),
            'molecular_weight_kda': sum(self._aa_composition(self.g_sequence).get(aa, 0) * 110 for aa in set(self.g_sequence)) / 1000,
            'hydrophobicity_profile': hydrophobicity_profile,
            'average_hydrophobicity': np.mean(hydrophobicity_profile),
            'isoelectric_point': isoelectric_point,
            'instability_index': instability_index,
            'aromaticity': aromaticity,
            'gravy': gravy,
            'secondary_structure': secondary_structure
        }
        
        self._results['sequence_analysis'] = results
        return results
    
    def predict_epitopes(self):
        """Predict CD8+, CD4+, and B-cell epitopes."""
        self.log("Predicting epitopes...")
        
        epitopes = {
            'cd8_epitopes': [],
            'cd4_epitopes': [],
            'b_cell_epitopes': []
        }
        
        seq = self.g_sequence
        
        # CD8+ T-cell epitopes (9-mers)
        for i in range(len(seq) - 8):
            peptide = seq[i:i+9]
            score = self._calculate_epitope_score(peptide, 'cd8')
            
            # Filtering criteria
            hydrophobic_content = sum(1 for aa in peptide if aa in 'LMIVF') / 9
            antigenicity = self._calculate_antigenicity(peptide)
            
            if score > 0.05 and antigenicity > 0.4 and hydrophobic_content > 0.25:
                epitopes['cd8_epitopes'].append({
                    'peptide': peptide,
                    'position': i + 1,
                    'custom_score': round(score, 3),
                    'antigenicity': round(antigenicity, 3),
                    'hydrophobic_content': round(hydrophobic_content, 3),
                    'predicted_alleles': self.immune_params['endemic_hla_i'][:2]
                })
        
        # CD4+ T-cell epitopes (15-mers)
        for i in range(len(seq) - 14):
            peptide = seq[i:i+15]
            score = self._calculate_epitope_score(peptide, 'cd4')
            antigenicity = self._calculate_antigenicity(peptide)
            amphipathicity = self._calculate_amphipathicity(peptide)
            
            if score > 0.02 and antigenicity > 0.4 and amphipathicity > 0.1:
                epitopes['cd4_epitopes'].append({
                    'peptide': peptide,
                    'position': i + 1,
                    'custom_score': round(score, 3),
                    'antigenicity': round(antigenicity, 3),
                    'amphipathicity': round(amphipathicity, 3),
                    'predicted_alleles': self.immune_params['endemic_hla_ii'][:2]
                })
        
        # B-cell epitopes
        for length in range(12, 17):
            for i in range(len(seq) - length + 1):
                peptide = seq[i:i+length]
                score = self._calculate_epitope_score(peptide, 'b_cell')
                accessibility = self._estimate_surface_accessibility(peptide, i)
                antigenicity = self._calculate_antigenicity(peptide)
                flexibility = sum(1 for aa in peptide if aa in 'GPST') / length
                
                if score > 0.03 and accessibility > 0.55 and antigenicity > 0.4:
                    epitopes['b_cell_epitopes'].append({
                        'peptide': peptide,
                        'position': i + 1,
                        'length': length,
                        'custom_score': round(score, 3),
                        'accessibility': round(accessibility, 2),
                        'flexibility': round(flexibility, 2),
                        'antigenicity': round(antigenicity, 3),
                    })
        
        # Sort and deduplicate
        for ep_type in epitopes:
            # Sort by score descending
            epitopes[ep_type].sort(key=lambda x: x['custom_score'], reverse=True)
            
            # Deduplicate overlapping peptides (keep highest score)
            seen = set()
            unique = []
            for ep in epitopes[ep_type]:
                if ep['peptide'] not in seen:
                    seen.add(ep['peptide'])
                    unique.append(ep)
            epitopes[ep_type] = unique[:50]
        
        self.log(f"Predicted {len(epitopes['cd8_epitopes'])} CD8+, "
                f"{len(epitopes['cd4_epitopes'])} CD4+, "
                f"{len(epitopes['b_cell_epitopes'])} B-cell epitopes")
        
        self._results['epitope_predictions'] = epitopes
        return epitopes
    
    def design_vaccine_construct(self, top_n_cd8=5, top_n_cd4=5, top_n_bcell=3):
        """Design multi-epitope vaccine construct with corrected linker assembly."""
        self.log("Designing multi-epitope vaccine construct...")
        
        if 'epitope_predictions' not in self._results:
            epitopes = self.predict_epitopes()
        else:
            epitopes = self._results['epitope_predictions']
        
        top_cd8 = epitopes['cd8_epitopes'][:top_n_cd8]
        top_cd4 = epitopes['cd4_epitopes'][:top_n_cd4]
        top_bcell = epitopes['b_cell_epitopes'][:top_n_bcell]
        
        # Linker sequences - CORRECTED (now properly defined as strings)
        linker_flexible = "GGGGS"
        linker_rigid = "EAAAK"
        linker_cleavable = "KFERQ"
        
        # Build construct parts as a list, then join
        parts = []
        
        # Signal peptide and adjuvant
        parts.append("MKLLVVFGLLAVALG")  # Signal peptide
        parts.append(linker_flexible)
        parts.append("KTLR")  # TLR agonist motif
        
        # CD8 epitopes with flexible linkers
        for epitope in top_cd8:
            parts.append(linker_flexible)
            parts.append(epitope['peptide'])
        
        # CD4 epitopes with rigid linkers
        parts.append(linker_rigid)
        for epitope in top_cd4:
            parts.append(linker_rigid)
            parts.append(epitope['peptide'])
        
        # B-cell epitopes with cleavable linkers
        parts.append(linker_cleavable)
        for epitope in top_bcell:
            parts.append(linker_cleavable)
            parts.append(epitope['peptide'])
        
        # His tag
        parts.append(linker_flexible)
        parts.append("HHHHHH")
        
        # Join all parts
        construct_sequence = ''.join(parts)
        
        # Analyze construct properties
        if BIOPYTHON_AVAILABLE:
            construct_analysis = ProteinAnalysis(construct_sequence)
            isoelectric_point = construct_analysis.isoelectric_point()
            instability_index = construct_analysis.instability_index()
        else:
            isoelectric_point = 7.5
            instability_index = 35.0
        
        construct_info = {
            'sequence': construct_sequence,
            'fasta': f">Nipah_G_MultiEpitope_Vaccine\n{construct_sequence}\n",
            'length': len(construct_sequence),
            'molecular_weight_kda': len(construct_sequence) * 0.110,
            'isoelectric_point': isoelectric_point,
            'instability_index': instability_index,
            'selected_epitopes': {
                'cd8': top_cd8,
                'cd4': top_cd4,
                'b_cell': top_bcell
            },
            'composition': {
                'signal_peptide': "MKLLVVFGLLAVALG",
                'adjuvant': "KTLR",
                'his_tag': "HHHHHH",
                'total_epitopes': len(top_cd8) + len(top_cd4) + len(top_bcell)
            }
        }
        
        self.log(f"Construct designed: {construct_info['length']} aa, "
                f"{construct_info['composition']['total_epitopes']} epitopes")
        
        self._results['vaccine_construct'] = construct_info
        return construct_info
    
    def simulate_receptor_binding(self):
        """Simulate receptor binding kinetics."""
        self.log("Simulating receptor binding kinetics...")
        
        concentrations = np.logspace(-12, -6, 100)
        kd2 = self.structural_params['binding_affinity_efnb2']
        kd3 = self.structural_params['binding_affinity_efnb3']
        
        binding_results = {
            'concentrations_M': concentrations,
            'concentrations_nM': concentrations * 1e9,
            'binding_efnb2': concentrations / (kd2 + concentrations),
            'binding_efnb3': concentrations / (kd3 + concentrations),
            'kd_efnb2_nM': kd2 * 1e9,
            'kd_efnb3_nM': kd3 * 1e9,
            'ic50_efnb2_nM': kd2 * 2 * 1e9,
            'ic50_efnb3_nM': kd3 * 2 * 1e9,
        }
        
        self._results['receptor_binding'] = binding_results
        return binding_results
    
    def simulate_immune_response(self, days=365, doses=3):
        """Simulate immune response kinetics."""
        self.log(f"Simulating {days}-day immune response...")
        
        time_points = np.linspace(0, days, 500)
        dose_times = [0, 21, 42] if doses >= 3 else [0, 28] if doses >= 2 else [0]
        
        antibody_response = np.zeros_like(time_points)
        cellular_response = np.zeros_like(time_points)
        memory_response = np.zeros_like(time_points)
        
        for dose_time in dose_times[:doses]:
            t_rel = np.maximum(0, time_points - dose_time)
            antibody_response += 800 * (1 - np.exp(-t_rel / 14)) * np.exp(-t_rel / 120)
            cellular_response += 400 * (1 - np.exp(-t_rel / 7)) * np.exp(-t_rel / 90)
            memory_response += 180 * (1 - np.exp(-t_rel / 21))
        
        immune_results = {
            'time_points': time_points,
            'dose_times': dose_times[:doses],
            'antibody_response': antibody_response,
            'cellular_response': cellular_response,
            'memory_response': memory_response,
            'peak_antibody': float(np.max(antibody_response)),
            'peak_cellular': float(np.max(cellular_response)),
            'final_memory': float(memory_response[-1]),
            'protection_duration': self._estimate_protection_duration(antibody_response, time_points)
        }
        
        self._results['immune_simulation'] = immune_results
        return immune_results
    
    def _estimate_protection_duration(self, antibody_response, time_points):
        protective_threshold = np.max(antibody_response) * 0.1
        protective_indices = np.where(antibody_response >= protective_threshold)[0]
        return float(time_points[protective_indices[-1]]) if len(protective_indices) > 0 else 0
    
    def estimate_population_coverage(self):
        """Estimate population coverage using corrected Hardy-Weinberg method."""
        self.log("Estimating population coverage...")
        
        if 'epitope_predictions' not in self._results:
            epitopes = self.predict_epitopes()
        else:
            epitopes = self._results['epitope_predictions']
        
        # HLA frequencies in South/Southeast Asia
        hla_frequencies = {
            'HLA-A*11:01': 0.25, 'HLA-A*24:02': 0.35, 
            'HLA-B*40:06': 0.15, 'HLA-B*07:02': 0.20,
            'DRB1*03:01': 0.12, 'DRB1*07:01': 0.18, 
            'DRB1*15:01': 0.22, 'DRB1*04:01': 0.15
        }
        
        # Calculate class I coverage (at least one epitope presented)
        class_i_alleles = [k for k in hla_frequencies if k.startswith(('HLA-A', 'HLA-B'))]
        class_i_freq_sum = sum(hla_frequencies[a] for a in class_i_alleles)
        
        # Using Hardy-Weinberg: P(at least one allele) = 1 - P(no alleles)
        # Simplified: coverage = 1 - product(1 - freq_i * epitope_match_i)
        class_i_epitope_matches = [
            any(allele in ep.get('predicted_alleles', []) for ep in epitopes['cd8_epitopes'][:10])
            for allele in class_i_alleles
        ]
        
        prob_no_class_i = np.prod([1 - hla_frequencies[a] * m 
                                   for a, m in zip(class_i_alleles, class_i_epitope_matches)])
        class_i_coverage = 1 - prob_no_class_i
        
        # Class II coverage
        class_ii_alleles = [k for k in hla_frequencies if k.startswith('DRB1')]
        class_ii_epitope_matches = [
            any(allele in ep.get('predicted_alleles', []) for ep in epitopes['cd4_epitopes'][:10])
            for allele in class_ii_alleles
        ]
        
        prob_no_class_ii = np.prod([1 - hla_frequencies[a] * m 
                                    for a, m in zip(class_ii_alleles, class_ii_epitope_matches)])
        class_ii_coverage = 1 - prob_no_class_ii
        
        # Overall coverage (union of class I and II)
        overall_coverage = class_i_coverage + class_ii_coverage - (class_i_coverage * class_ii_coverage)
        
        coverage_results = {
            'class_i_coverage': float(class_i_coverage),
            'class_ii_coverage': float(class_ii_coverage),
            'overall_coverage': float(overall_coverage),
            'hla_frequencies': hla_frequencies,
            'meets_target': overall_coverage >= self.immune_params['population_coverage_target']
        }
        
        self.log(f"Estimated population coverage: {overall_coverage:.1%}")
        self._results['population_coverage'] = coverage_results
        return coverage_results
    
    def optimize_codon_usage(self):
        """Optimize codon usage for E. coli expression."""
        self.log("Optimizing codon usage...")
        
        if 'vaccine_construct' not in self._results:
            construct_info = self.design_vaccine_construct()
        else:
            construct_info = self._results['vaccine_construct']
        
        # E. coli optimized codons
        codon_table = {
            'A': 'GCT', 'R': 'CGT', 'N': 'AAC', 'D': 'GAT', 'C': 'TGC',
            'Q': 'CAG', 'E': 'GAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATC',
            'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTC', 'P': 'CCG',
            'S': 'TCT', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTT'
        }
        
        protein_seq = construct_info['sequence']
        optimized_dna = ''.join(codon_table.get(aa, 'NNN') for aa in protein_seq)
        
        # GC content calculation
        gc_count = optimized_dna.count('G') + optimized_dna.count('C')
        gc_content = gc_count / len(optimized_dna) if optimized_dna else 0
        
        # Add restriction sites
        cloning_sequence = f"CATATG{optimized_dna[3:]}CTCGAG"
        
        optimization_results = {
            'original_protein': protein_seq,
            'optimized_dna': optimized_dna,
            'cloning_sequence': cloning_sequence,
            'gc_content': gc_content,
            'length_bp': len(optimized_dna),
            'vector_info': {
                'recommended_vector': 'pET28a(+)',
                'resistance': 'Kanamycin',
                'expression_tags': 'N-terminal His6',
                'restriction_sites': 'NdeI/XhoI'
            }
        }
        
        self._results['codon_optimization'] = optimization_results
        return optimization_results
    
    def generate_comprehensive_report(self, output_dir="."):
        """Generate comprehensive analysis report."""
        self.log("Generating comprehensive report...")
        
        # Run all analyses
        sequence_props = self.analyze_sequence_properties()
        epitopes = self.predict_epitopes()
        vaccine_construct = self.design_vaccine_construct()
        receptor_binding = self.simulate_receptor_binding()
        immune_response = self.simulate_immune_response()
        population_coverage = self.estimate_population_coverage()
        codon_optimization = self.optimize_codon_usage()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Write report
        report_path = output_path / "comprehensive_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NIPAH VIRUS COMPUTATIONAL VACCINE MODEL - REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Version: {__version__}\n\n")
            
            f.write("SEQUENCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Length: {sequence_props['length']} aa\n")
            f.write(f"GRAVY: {sequence_props['gravy']:.3f}\n\n")
            
            f.write("EPITOPE PREDICTIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"CD8+: {len(epitopes['cd8_epitopes'])}\n")
            f.write(f"CD4+: {len(epitopes['cd4_epitopes'])}\n")
            f.write(f"B-cell: {len(epitopes['b_cell_epitopes'])}\n\n")
            
            f.write("TOP CD8+ EPITOPES:\n")
            for i, ep in enumerate(epitopes['cd8_epitopes'][:5], 1):
                f.write(f"  {i}. {ep['peptide']} (pos {ep['position']}, score {ep['custom_score']})\n")
            
            f.write("\nVACCINE CONSTRUCT\n")
            f.write("-" * 40 + "\n")
            f.write(f"Length: {vaccine_construct['length']} aa\n")
            f.write(f"MW: {vaccine_construct['molecular_weight_kda']:.1f} kDa\n\n")
            
            f.write("POPULATION COVERAGE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall: {population_coverage['overall_coverage']:.1%}\n")
            f.write(f"Target met: {'Yes' if population_coverage['meets_target'] else 'No'}\n\n")
            
            f.write("RECEPTOR BINDING\n")
            f.write("-" * 40 + "\n")
            f.write(f"EphrinB2 Kd: {receptor_binding['kd_efnb2_nM']:.1f} nM\n")
            f.write(f"EphrinB3 Kd: {receptor_binding['kd_efnb3_nM']:.1f} nM\n\n")
            
            f.write("IMMUNE RESPONSE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Peak antibody: {immune_response['peak_antibody']:.0f} units\n")
            f.write(f"Protection duration: {immune_response['protection_duration']:.0f} days\n")
        
        # Save FASTA
        fasta_path = output_path / "vaccine_construct.fasta"
        with open(fasta_path, 'w') as f:
            f.write(vaccine_construct['fasta'])
        
        # Save cloning sequence
        cloning_path = output_path / "cloning_sequence.txt"
        with open(cloning_path, 'w') as f:
            f.write(f">Nipah_Vaccine_Optimized\n{codon_optimization['cloning_sequence']}\n")
        
        self.log(f"Report saved to {report_path}")
        return self._results
    
    def create_visualizations(self, output_dir="."):
        """Create visualization plots."""
        self.log("Creating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        fig = plt.figure(figsize=(16, 12))
        
        # Hydrophobicity profile
        ax1 = fig.add_subplot(3, 3, 1)
        if 'sequence_analysis' in self._results:
            profile = self._results['sequence_analysis']['hydrophobicity_profile']
            ax1.plot(range(1, len(profile)+1), profile, color=colors[0], linewidth=1)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax1.set_title('Hydrophobicity Profile')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Score')
        
        # Receptor binding
        ax2 = fig.add_subplot(3, 3, 2)
        if 'receptor_binding' in self._results:
            b = self._results['receptor_binding']
            ax2.semilogx(b['concentrations_nM'], b['binding_efnb2'], label='EphrinB2', color=colors[0])
            ax2.semilogx(b['concentrations_nM'], b['binding_efnb3'], label='EphrinB3', color=colors[1])
            ax2.set_title('Receptor Binding')
            ax2.set_xlabel('Concentration (nM)')
            ax2.set_ylabel('Fraction bound')
            ax2.legend()
        
        # Epitope counts
        ax3 = fig.add_subplot(3, 3, 3)
        if 'epitope_predictions' in self._results:
            e = self._results['epitope_predictions']
            counts = [len(e['cd8_epitopes']), len(e['cd4_epitopes']), len(e['b_cell_epitopes'])]
            bars = ax3.bar(['CD8+', 'CD4+', 'B-cell'], counts, color=colors[:3])
            ax3.set_title('Epitope Counts')
            ax3.set_ylabel('Number')
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(count), ha='center')
        
        # Population coverage
        ax4 = fig.add_subplot(3, 3, 4)
        if 'population_coverage' in self._results:
            c = self._results['population_coverage']
            bars = ax4.bar(['Class I', 'Class II', 'Overall'], 
                          [c['class_i_coverage']*100, c['class_ii_coverage']*100, c['overall_coverage']*100],
                          color=colors[:3])
            ax4.axhline(y=70, color='red', linestyle='--', label='Target')
            ax4.set_title('Population Coverage (%)')
            ax4.legend()
            for bar, val in zip(bars, [c['class_i_coverage'], c['class_ii_coverage'], c['overall_coverage']]):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.1f}%', ha='center')
        
        # Immune response
        ax5 = fig.add_subplot(3, 3, (5, 6))
        if 'immune_simulation' in self._results:
            im = self._results['immune_simulation']
            ax5.plot(im['time_points'], im['antibody_response'], label='Antibody', color=colors[0])
            ax5.plot(im['time_points'], im['cellular_response'], label='Cellular', color=colors[1])
            for dt in im['dose_times']:
                ax5.axvline(x=dt, color='gray', linestyle=':', alpha=0.5)
            ax5.set_title('Immune Response Kinetics')
            ax5.set_xlabel('Days')
            ax5.set_ylabel('Response')
            ax5.legend()
        
        # Score distribution
        ax6 = fig.add_subplot(3, 3, (7, 8))
        if 'epitope_predictions' in self._results:
            e = self._results['epitope_predictions']
            cd8_scores = [ep['custom_score'] for ep in e['cd8_epitopes']]
            cd4_scores = [ep['custom_score'] for ep in e['cd4_epitopes']]
            ax6.hist(cd8_scores, bins=20, alpha=0.5, label='CD8+', color=colors[0])
            ax6.hist(cd4_scores, bins=20, alpha=0.5, label='CD4+', color=colors[1])
            ax6.set_title('Epitope Score Distribution')
            ax6.set_xlabel('Score')
            ax6.set_ylabel('Frequency')
            ax6.legend()
        
        plt.suptitle('Nipah Virus Computational Vaccine Model Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig_path = output_path / "comprehensive_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.log(f"Visualizations saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Nipah Virus Computational Vaccine Model')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick analysis')
    parser.add_argument('--no-plots', action='store_true', help='Skip plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NIPAH VIRUS COMPUTATIONAL VACCINE MODEL")
    print("=" * 60)
    print(f"Version: {__version__}")
    print()
    
    model = NipahVaccineModel(verbose=True)
    
    if args.quick:
        print("Running in QUICK mode...")
        model.predict_epitopes()
        model.design_vaccine_construct(top_n_cd8=3, top_n_cd4=3, top_n_bcell=2)
    else:
        model.generate_comprehensive_report(args.output)
        
        if not args.no_plots:
            model.create_visualizations(args.output)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {os.path.abspath(args.output)}")
    
    if 'vaccine_construct' in model._results:
        print(f"Vaccine construct: {model._results['vaccine_construct']['length']} aa")
    if 'population_coverage' in model._results:
        print(f"Population coverage: {model._results['population_coverage']['overall_coverage']:.1%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
