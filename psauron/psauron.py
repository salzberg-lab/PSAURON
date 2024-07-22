import os
import sys
import gzip
import argparse
import warnings
import numpy as np
from tqdm.auto import tqdm

import pkg_resources

from scipy.special import expit, logit

import torch
import torch.nn.functional as F

# from multiprocessing import Pool
# import subprocess

from psauron.TCN_model import TCN, tokenize_aa_seq

trans_table = {
    "AAA": "K", "AAC": "N", "AAG": "K", "AAT": "N", "AAR": "K", "AAY": "N", "ACA": "T", "ACC": "T",
    "ACG": "T", "ACT": "T", "ACR": "T", "ACY": "T", "ACS": "T", "ACW": "T", "ACK": "T", "ACM": "T",
    "ACB": "T", "ACD": "T", "ACH": "T", "ACV": "T", "ACN": "T", "AGA": "R", "AGC": "S", "AGG": "R",
    "AGT": "S", "AGR": "R", "AGY": "S", "ATA": "I", "ATC": "I", "ATG": "M", "ATT": "I", "ATY": "I",
    "ATW": "I", "ATM": "I", "ATH": "I", "CAA": "Q", "CAC": "H", "CAG": "Q", "CAT": "H", "CAR": "Q", 
    "CAY": "H", "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P", "CCR": "P", "CCY": "P", "CCS": "P", 
    "CCW": "P", "CCK": "P", "CCM": "P", "CCB": "P", "CCD": "P", "CCH": "P", "CCV": "P", "CCN": "P", 
    "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R", "CGR": "R", "CGY": "R", "CGS": "R", "CGW": "R", 
    "CGK": "R", "CGM": "R", "CGB": "R", "CGD": "R", "CGH": "R", "CGV": "R", "CGN": "R", "CTA": "L", 
    "CTC": "L", "CTG": "L", "CTT": "L", "CTR": "L", "CTY": "L", "CTS": "L", "CTW": "L", "CTK": "L", 
    "CTM": "L", "CTB": "L", "CTD": "L", "CTH": "L", "CTV": "L", "CTN": "L", "GAA": "E", "GAC": "D", 
    "GAG": "E", "GAT": "D", "GAR": "E", "GAY": "D", "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A", 
    "GCR": "A", "GCY": "A", "GCS": "A", "GCW": "A", "GCK": "A", "GCM": "A", "GCB": "A", "GCD": "A", 
    "GCH": "A", "GCV": "A", "GCN": "A", "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G", "GGR": "G", 
    "GGY": "G", "GGS": "G", "GGW": "G", "GGK": "G", "GGM": "G", "GGB": "G", "GGD": "G", "GGH": "G", 
    "GGV": "G", "GGN": "G", "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V", "GTR": "V", "GTY": "V", 
    "GTS": "V", "GTW": "V", "GTK": "V", "GTM": "V", "GTB": "V", "GTD": "V", "GTH": "V", "GTV": "V", 
    "GTN": "V", "TAA": "*", "TAC": "Y", "TAG": "*", "TAT": "Y", "TAR": "*", "TAY": "Y", "TCA": "S", 
    "TCC": "S", "TCG": "S", "TCT": "S", "TCR": "S", "TCY": "S", "TCS": "S", "TCW": "S", "TCK": "S", 
    "TCM": "S", "TCB": "S", "TCD": "S", "TCH": "S", "TCV": "S", "TCN": "S", "TGA": "*", "TGC": "C", 
    "TGG": "W", "TGT": "C", "TGY": "C", "TTA": "L", "TTC": "F", "TTG": "L", "TTT": "F", "TTR": "L", 
    "TTY": "F", "TRA": "*", "YTA": "L", "YTG": "L", "YTR": "L", "MGA": "R", "MGG": "R", "MGR": "R"
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-fasta", type=str, required=True, help="REQUIRED path to FASTA with spliced CDS sequence or protein sequence. A spliced CDS fasta can be created from a GTF/GFF and a reference FASTA by using gffread.")
    parser.add_argument("-o", "--output-path", type=str, required=False, help="OPTIONAL path to output results file, default=./psauron_score.csv", default="./psauron_score.csv")
    parser.add_argument("-m", "--minimum-length", type=int, required=False, help="OPTIONAL exclude all proteins shorter than m amino acids, default=5", default=5)
    parser.add_argument("-e", "--exclude", type=str, help="OPTIONAL exclude any CDS where FASTA description contains given text (case invariant), e.g. \"hypothetical\", default=None", default="")
    parser.add_argument("--inframe", type=float, required=False, help="OPTIONAL probability threshold used to determine final psauron score, in-frame, higher number decreases sensitivity and increases specificity, default=0.5, range=[0,1]", default=0.5)
    parser.add_argument("--outframe", type=float, required=False, help="OPTIONAL probability threshold used to determine final psauron score, out-of-frame, higher number increases sensitivity and decreases specificity, default=0.5, range=[0,1]", default=0.5)
    parser.add_argument("-c", "--use-cpu", action='store_true', help="OPTIONAL set -c to force usage of CPU instead of GPU, default=False", default=False)
    parser.add_argument("-s", "--single-frame", action='store_true', help="OPTIONAL set -s to score only the in-frame CDS, which may lower accuracy of the model, default=False", default=False)
    parser.add_argument("-p", "--protein", action='store_true', help="OPTIONAL set -p if your FASTA contains amino acid protein sequence, which may lower accuracy of the model, default=False", default=False)
    parser.add_argument("-a", "--all-prob", action='store_true', help="OPTIONAL set -a to output per-amino-acid predicted probabilities, NOTE: these may not behave as expected due to receptive field size, default=False", default=False)
    parser.add_argument("-v", "--verbose", action='store_true', help="OPTIONAL set -v for verbose output with progress bars etc., default=False", default=False)

    args = parser.parse_args()
    return args
    
def get_data_path():
    # gets correct path to model weight data for installed package
    return pkg_resources.resource_filename(__name__, 'data/model_state_dict.pt')
    
def load_model(use_cpu):
    # load TCN model
    input_channels = 21
    n_classes = 1
    kernel_size = 16
    dropout = 0.05
    hidden_units_per_layer = 32
    levels = 6
    
    checkpoint = get_data_path()

    if torch.cuda.device_count() > 0 and not use_cpu:
        state_dict = torch.load(checkpoint)
    else:
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        
    channel_sizes = [hidden_units_per_layer] * levels
    model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 0 and not use_cpu:
        model.to('cuda')
    model.eval()
    
    return model
    
def reverse_complement(dna):
    # reverse complements DNA seq, returns A for all chars not in dict
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join([complement.get(base, 'A') for base in dna[::-1]])

def translate(sequence):
    dna_sequence = sequence.upper().strip()
    # truncate to last full codon
    length = len(dna_sequence)
    if length < 3:
        return ""
    dna_sequence = dna_sequence[:length-(length % 3)]
    dna_sequence = dna_sequence.replace('U', 'T') # replaces U with T in case of RNA

    if set(dna_sequence) - {"A", "C", "G", "T", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N"}: 
        raise ValueError(f"Sequence {dna_sequence} contains illegal letters")

    protein_sequence = []

    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i+3]
        amino_acid = trans_table.get(codon, 'X')
        protein_sequence.append(amino_acid)

    protein_string = ''.join(protein_sequence)

    return protein_string

def load_fasta(filepath):
    '''Loads a FASTA file and returns a list of descriptions and a list of sequences'''
    print("Loading transcripts at", filepath)
    if filepath[-3:].lower() == ".gz":
        f = gzip.open(filepath, "rt")
    else:
        f = open(filepath, "rt")
    
    description_list = []
    seq_list = []
    for name, seq, qual in readfq(f):
        description_list.append(name)
        seq_list.append(seq)
    
    f.close()
    print("Loaded...\n")
    return description_list, seq_list
    
def readfq(fp): 
    """this is a generator function copied from Heng Li's readfq project https://github.com/lh3/readfq/blob/master/readfq.py"""
    last = None # this is a buffer keeping the last unprocessed line
    while True: # mimic closure; is it a bad idea?
        if not last: # the first record or a record following a fastq
            for l in fp: # search for the start of the next record
                if l[0] in '>@': # fasta/q header line
                    last = l[:-1] # save this line
                    break
        if not last: break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp: # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+': # this is a fasta record
            yield name, ''.join(seqs), None # yield a fasta record
            if not last: break
        else: # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp: # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq): # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs); # yield a fastq record
                    break
            if last: # reach EOF before reading enough quality
                yield name, seq, None # yield a fasta record instead
                break
    
def predict(X, model, use_cpu):
    model.eval()
    with torch.no_grad():
        if torch.cuda.device_count() > 0 and not use_cpu:
            X_enc = F.one_hot(X, 21).permute(0,2,1).float().cuda()
            probs = expit(model(X_enc).cpu())
            del X_enc
            torch.cuda.empty_cache()
        else:
            X_enc = F.one_hot(X, 21).permute(0,2,1).float()
            probs = expit(model(X_enc).cpu())

    return probs
    
def score_seq(aa_seq_list, model, use_cpu, allprob, verbose, gene_batch_size=100):
    # format seqs
    ORF_seq_enc = [tokenize_aa_seq(str(x)) for x in aa_seq_list]

    # sort by length to minimize impact of batch padding
    ORF_lengths = np.asarray([len(x) for x in ORF_seq_enc])
    length_idx = np.argsort(ORF_lengths)
    ORF_seq_sorted = [ORF_seq_enc[i] for i in length_idx]

    # pad to allow creation of batch matrix
    prob_list = []
    prob_list_allprob = []
    if verbose:
        for i in tqdm(range(0, len(ORF_seq_sorted), gene_batch_size), unit=" batch"):
            batch = ORF_seq_sorted[i:i+gene_batch_size]
            seq_lengths = torch.LongTensor(list(map(len, batch)))
            seq_tensor = torch.zeros((len(batch), seq_lengths.max())).long()
            for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            pred_all = predict(seq_tensor, model, use_cpu)
            pred = []
            pred_allprob = []
            for j, length in enumerate(seq_lengths):
                subseq = pred_all[j, 0, 0:int(length)]
                idx = min(100, int(length) - 1) # avoids NaNs
                predprob = float(expit(torch.mean(logit(subseq[idx:]))))
                pred.append(predprob)
                pred_allprob.append(subseq.tolist())
            prob_list.extend(pred)
            prob_list_allprob.extend(pred_allprob)
        prob_arr = np.asarray(prob_list, dtype=float)
    else:
        for i in range(0, len(ORF_seq_sorted), gene_batch_size):
            batch = ORF_seq_sorted[i:i+gene_batch_size]
            seq_lengths = torch.LongTensor(list(map(len, batch)))
            seq_tensor = torch.zeros((len(batch), seq_lengths.max())).long()
            for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            pred_all = predict(seq_tensor, model, use_cpu)
            pred = []
            pred_allprob = []
            for j, length in enumerate(seq_lengths):
                subseq = pred_all[j, 0, 0:int(length)]
                idx = min(100, int(length) - 1) # avoids NaNs
                predprob = float(expit(torch.mean(logit(subseq[idx:]))))
                pred.append(predprob)
                pred_allprob.append(subseq.tolist())
            prob_list.extend(pred)
            prob_list_allprob.extend(pred_allprob)
        prob_arr = np.asarray(prob_list, dtype=float)

    # unsort
    unsort_idx = np.argsort(length_idx)
    ORF_prob = prob_arr[unsort_idx]
    
    # unsort allprob, differing lengths mean this can't be a np array
    ORF_prob_allprob = []
    if allprob:
        for idx in unsort_idx:
            ORF_prob_allprob.append(prob_list_allprob[idx])
    
    return ORF_prob, ORF_prob_allprob
    
def eye_of_psauron():
    # supress annoying warnings
    warnings.filterwarnings('ignore')
    
    # print PSAURON version
    version = "1.0.4"
    print("PSAURON version", version)
    
    # parse command line arguments
    try:
        args = get_args()
    except:
        print("\n -i INPUT_FASTA, REQUIRED path to FASTA with spliced CDS sequence. This fasta can be created from a GTF/GFF and a reference FASTA by using gffread. \n\nExample gffread commands to get CDS FASTA:\ngffread -x CDS_FASTA.fa -g genome.fa input.gff\ngffread -x CDS_FASTA.fa -g genome.fa input.gtf\n")
        sys.exit()
    min_len_aa = args.minimum_length
    p_fasta = args.input_fasta
    use_cpu = args.use_cpu
    allprob = args.all_prob
    verbose = args.verbose
    single_frame = args.single_frame
    protein = args.protein
    print("System info")
    print(sys.version, "\n")
    
    # check args
    if single_frame and protein:
        print("cannot set both -s and -p, type psauron -h for detailed help documentation.")
        sys.exit()
    # TODO check correctness of file contents vs. -p setting. currently does not warn when run with -p on CDS.
    
    # i did my best
    if verbose:
        print(r"                                             _______________________")
        print(r"   _______________________-------------------                       `\\")
        print(r" /:--__                                                              |")
        print(r"||< > |                                   ___________________________/")
        print(r"| \__/_________________-------------------                          |")
        print(r" |                                                                  |")
        print(r" |       Three Chains for the Histones strong and spry,              |")
        print(r"  |        Seven for the Enzymes in their catalytic might,           |")
        print(r"  |            Nine for DNA, all it unifies,                         |")
        print(r"  |         One for the Program, the Genome's light.                 |")
        print(r"  |      In the Land of Annotation where the Proteins lie.           |")
        print(r"   |                                                                  |")
        print(r"   |       One Model to rule them all, One Model to find them,        |")
        print(r"   |      One Model to score them all and in the Paper cite them.     |")
        print(r"   |                                                                  |")
        print(r"  |                                              ____________________|_")
        print(r"  |  ___________________-------------------------                      `\\")
        print(r"  |/`--_                                                                 |")
        print(r"  ||[ ]||                                            ___________________/")
        print(r"   \===/___________________--------------------------")
        print("\n")
    else:
        print(r"                                             _______________________")
        print(r"   _______________________-------------------                       `\\")
        print(r" /:--__                                                              |")
        print(r"||< > |                                   ___________________________/")
        print(r"| \__/_________________-------------------                          |")
        print(r" |                                                                  |")
        print(r"  |                                                                  |")
        print(r"  |       One Model to rule them all, One Model to find them,        |")
        print(r"  |      One Model to score them all and in the Paper cite them.     |")
        print(r"   |                                                                  |")
        print(r"   |                                              ____________________|_")
        print(r"  |  ___________________-------------------------                      `\\")
        print(r"  |/`--_                                                                 |")
        print(r"  ||[ ]||                                            ___________________/")
        print(r"   \===/___________________--------------------------")
        print("\n")
        
    # load TCN model from locally installed data
    print("Loading TCN model...")
    model = load_model(use_cpu)
    print("Loaded...")
    
    # CDS SCORING
    if not args.protein:
        # load CDS from fasta
        print("Loading CDS FASTA...")
        description_list, seq_list = load_fasta(p_fasta)
        n_excluded = 0
        if args.exclude:
            new_description_list = []
            new_seq_list = []
            for desc, seq in zip(description_list, seq_list):
                if args.exclude.lower() not in desc.lower():
                    new_description_list.append(desc)
                    new_seq_list.append(seq)
                else:
                    n_excluded += 1
            description_list = new_description_list
            seq_list = new_seq_list
                    
        print("Loaded", str(len(seq_list)), "CDS sequences...")
        if len(args.exclude) > 0:
            print("Excluding", str(n_excluded), "sequences containing", "\"" + str(args.exclude) + "\"")
            
        # check data
        if len(seq_list) == 0:
            print("No data to score, please check data and/or arguments.")
            sys.exit()
        
        # uppercase and remove N's
        seq_list = [str(x).upper().replace("N", "A") for x in seq_list]
        
        # ALL FRAME SCORING, DEFAULT BEHAVIOR
        if not single_frame:
            # get alternate reading frame sequence
            ARF_list = []
            for s in seq_list:
                revcomp = reverse_complement(s)
                ARFs = [s[1:], s[2:], revcomp, revcomp[1:], revcomp[2:]]
                ARF_list.append(ARFs)

            # translate CDS to protein seq in all frames
            print("Translating CDS sequences in all reading frames...")
            ORF_list_aa = []
            ARF_list_aa = []
            if verbose:
                for i in tqdm(range(len(seq_list))):
                    ORF_list_aa.append(translate(seq_list[i]))
                    ARF_list_aa.append([translate(x) for x in ARF_list[i]])
            else:
                for i in range(len(seq_list)):
                    ORF_list_aa.append(translate(seq_list[i]))
                    ARF_list_aa.append([translate(x) for x in ARF_list[i]])
            
            # remove stop codons from alternate reading frames
            ORF_list_aa = [s.replace('*', '') for s in ORF_list_aa]
            ARF_list_aa = [[s.replace('*', '') for s in x] for x in ARF_list_aa]
                        
            # filter by minimum length
            ORF_list_aa_flat = ORF_list_aa
            ARF_list_aa_flat = [item for sublist in ARF_list_aa for item in sublist]          
            
            idx = [i for i in range(len(ORF_list_aa_flat)) if len(ORF_list_aa_flat[i]) >= min_len_aa]
            ORF_list_aa_flat_clean = []
            ARF_list_aa_flat_clean = []
            description_list_clean = []
            for i in idx:
                ORF_list_aa_flat_clean.append(ORF_list_aa_flat[i])
                ARF_list_aa_flat_clean.extend(ARF_list_aa[i])
                description_list_clean.append(description_list[i])               
            
            print("Excluding", str(len(seq_list) - len(idx)), "proteins with length below m amino acids from analysis, m =", str(min_len_aa), "\n")
            if len(idx) == 0:
                print("No data to score, please check data and/or arguments.")
                sys.exit()
            
            # score all frames with model
            if torch.cuda.device_count() > 0:
                if not use_cpu:
                    print("Running TCN model on GPU...")
                else:
                    print("Detected GPU, but running TCN model on CPU due to -c argument")
            else:
                print("No GPU detected, running TCN model on CPU...")
            print("Scoring in-frame sequence...")
            ORF_prob, ORF_prob_allprob = score_seq(ORF_list_aa_flat_clean, model, use_cpu, allprob, verbose)
            print("Scoring out-of-frame sequence...")
            ARF_prob, ARF_prob_allprob = score_seq(ARF_list_aa_flat_clean, model, use_cpu, allprob, verbose)
            ARF_prob = [ARF_prob[i:i+5] for i in range(0, len(ARF_prob), 5)]    
            ARF_prob_allprob = [ARF_prob_allprob[i:i+5] for i in range(0, len(ARF_prob_allprob), 5)]
            
            # create meta-score for each CDS
            ORF_bound = args.inframe # ORF must be >= ORF_bound
            ARF_bound = args.outframe # mean ARFs must be <= ARF_bound
            is_protein = []
            ORF_score_all = []
            ARF_score_all = []
            ARF_score_mean_all = []
            for i in range(0, len(ORF_prob)):
                ORF_score = ORF_prob[i]
                ORF_score_all.append(ORF_score)
                ARF_score = [x for x in ARF_prob[i]]
                ARF_score_mean = np.mean(ARF_score)
                ARF_score_mean_all.append(ARF_score_mean)
                ARF_score_all.append(ARF_score)

                if ORF_score >= ORF_bound and ARF_score_mean <= ARF_bound:
                    is_protein.append(True)
                else:
                    is_protein.append(False)
                    
        
        # SINGLE FRAME SCORING, OPTIONAL BEHAVIOR
        else:
            # translate CDS to protein seq
            print("Translating CDS sequences in single CDS reading frame...")
            ORF_list_aa = []
            if verbose:
                for i in tqdm(range(len(seq_list))):
                    ORF_list_aa.append(translate(seq_list[i]))
            else:
                for i in range(len(seq_list)):
                    ORF_list_aa.append(translate(seq_list[i]))
                    
            # remove stop codons from alternate reading frames
            ORF_list_aa = [s.replace('*', '') for s in ORF_list_aa]
            
            # filter by minimum length
            ORF_list_aa_flat = ORF_list_aa
            idx = [i for i in range(len(ORF_list_aa_flat)) if len(ORF_list_aa_flat[i]) >= min_len_aa]
            ORF_list_aa_flat_clean = []
            description_list_clean = []
            for i in idx:
                ORF_list_aa_flat_clean.append(ORF_list_aa_flat[i])
                description_list_clean.append(description_list[i])
            print("Excluding", str(len(seq_list) - len(idx)), "proteins with length below m amino acids from analysis, m =", str(min_len_aa), "\n")
            if len(idx) == 0:
                print("No data to score, please check data and/or arguments.")
                sys.exit()
            
            # score single frame with model
            if torch.cuda.device_count() > 0:
                if not use_cpu:
                    print("Running TCN model on GPU...")
                else:
                    print("Detected GPU, but running TCN model on CPU due to -c argument")
            else:
                print("No GPU detected, running TCN model on CPU...")
            print("Scoring in-frame sequence...")
            ORF_prob, ORF_prob_allprob = score_seq(ORF_list_aa_flat_clean, model, use_cpu, allprob, verbose)
            
            # create score for each CDS
            ORF_bound = args.inframe # ORF must be >= ORF_bound
            is_protein = []
            ORF_score_all = []
            for i in range(0, len(ORF_prob)):
                ORF_score = ORF_prob[i]
                ORF_score_all.append(ORF_score)
                if ORF_score >= ORF_bound:
                    is_protein.append(True)
                else:
                    is_protein.append(False)

    
    # PROTEIN SCORING, OPTIONAL BEHAVIOR
    else:
        # load protein from fasta
        print("Loading protein FASTA...")
        description_list, seq_list = load_fasta(p_fasta)
        n_excluded = 0
        if args.exclude:
            new_description_list = []
            new_seq_list = []
            for desc, seq in zip(description_list, seq_list):
                if args.exclude.lower() not in desc.lower():
                    new_description_list.append(desc)
                    new_seq_list.append(seq)
                else:
                    n_excluded += 1
            description_list = new_description_list
            seq_list = new_seq_list
                    
                
        print("Loaded", str(len(seq_list)), "protein seqences...")
        if len(args.exclude) > 0:
            print("Excluding", str(n_excluded), "sequences containing", "\"" + str(args.exclude) + "\"")

        # check data
        if len(seq_list) == 0:
            print("No data to score, please check data and/or arguments.")
            sys.exit()
            
        # uppercase and remove stop codons
        ORF_list_aa = [str(x).upper() for x in seq_list]
        ORF_list_aa = [s.replace('*', '') for s in ORF_list_aa]
        
        # filter by minimum length
        ORF_list_aa_flat = ORF_list_aa
        idx = [i for i in range(len(ORF_list_aa_flat)) if len(ORF_list_aa_flat[i]) >= min_len_aa]
        ORF_list_aa_flat_clean = []
        description_list_clean = []
        for i in idx:
            ORF_list_aa_flat_clean.append(ORF_list_aa_flat[i])
            description_list_clean.append(description_list[i])
        print("Excluding", str(len(seq_list) - len(idx)), "proteins with length below m amino acids from analysis, m =", str(min_len_aa), "\n")
        if len(idx) == 0:
            print("No data to score, please check data and/or arguments.")
            sys.exit()
        
        # score single frame with model
        if torch.cuda.device_count() > 0:
            if not use_cpu:
                print("Running TCN model on GPU...")
            else:
                print("Detected GPU, but running TCN model on CPU due to -c argument")
        else:
            print("No GPU detected, running TCN model on CPU...")
        print("Scoring in-frame sequence...")
        ORF_prob, ORF_prob_allprob = score_seq(ORF_list_aa_flat_clean, model, use_cpu, allprob, verbose)
        
        # create score for each CDS
        ORF_bound = args.inframe # ORF must be >= ORF_bound
        is_protein = []
        ORF_score_all = []
        for i in range(0, len(ORF_prob)):
            ORF_score = ORF_prob[i]
            ORF_score_all.append(ORF_score)
            if ORF_score >= ORF_bound:
                is_protein.append(True)
            else:
                is_protein.append(False)

    
    # report results
    p_out = os.path.join(args.output_path)
    # print final score
    ss = round(sum(is_protein)/len(is_protein) * 100, 1)
    
    # ALL FRAMES, DEFAULT BEHAVIOR
    if not single_frame and not protein:
        print("\npsauron score:", ss)
        print("\nWriting detailed output file to", p_out)
        if allprob:
            print("NOTE: all_prob cells in .csv output may exceed Microsoft Excel max cell size limit")

        cols = ["description", "psauron_is_protein", "in_frame_score",
                "forward_frame2_score", "forward_frame3_score", 
                "reverse_frame1_score", "reverse_frame2_score", 
                "reverse_frame3_score", "mean_out_of_frame_score"]
        if allprob:
            cols.extend(["in_frame_all_prob", "forward_frame2_all_prob",
                         "forward_frame3_all_prob", "reverse_frame1_all_prob",
                         "reverse_frame2_all_prob", "reverse_frame3_all_prob"])   
             
        header = " ".join(sys.argv) + "\n" + "psauron score: " + str(ss) + "\nNOTE: alternate reading frames may not contain valid ORFs"
        with open(p_out, "wt") as f:
            f.write(header + "\n")
            f.write(",".join(cols) + "\n")
            for i, desc in enumerate(description_list_clean):
                if allprob:
                    # remove all "," in data to avoid .csv errors
                    f.write(",".join([str(x).replace(",", "") for x in [desc, 
                                                                        is_protein[i],
                                                                        round(ORF_score_all[i], 5),
                                                                        round(ARF_score_all[i][0], 5),
                                                                        round(ARF_score_all[i][1], 5),
                                                                        round(ARF_score_all[i][2], 5),
                                                                        round(ARF_score_all[i][3], 5),
                                                                        round(ARF_score_all[i][4], 5),
                                                                        round(ARF_score_mean_all[i], 5),
                                                                        ";".join([str(round(x,2)) for x in ORF_prob_allprob[i]]),
                                                                        ";".join([str(round(x,2)) for x in ARF_prob_allprob[i][0]]),
                                                                        ";".join([str(round(x,2)) for x in ARF_prob_allprob[i][1]]),
                                                                        ";".join([str(round(x,2)) for x in ARF_prob_allprob[i][2]]),
                                                                        ";".join([str(round(x,2)) for x in ARF_prob_allprob[i][3]]),
                                                                        ";".join([str(round(x,2)) for x in ARF_prob_allprob[i][4]]),
                                                                        ]]))
                    f.write("\n")
                else:
                    # remove all "," in data to avoid .csv errors
                    f.write(",".join([str(x).replace(",", "") for x in [desc, 
                                                                        is_protein[i],
                                                                        round(ORF_score_all[i], 5),
                                                                        round(ARF_score_all[i][0], 5),
                                                                        round(ARF_score_all[i][1], 5),
                                                                        round(ARF_score_all[i][2], 5),
                                                                        round(ARF_score_all[i][3], 5),
                                                                        round(ARF_score_all[i][4], 5),
                                                                        round(ARF_score_mean_all[i], 5)]]))
                    f.write("\n")
    
    else:
        print("\npsauron score:", ss)
        print("\nWriting detailed output file to", p_out)
        if allprob:
            print("NOTE: all_prob cells in .csv output may exceed Microsoft Excel max cell size limit")
        
        cols = ["description", "psauron_is_protein", "in-frame_score"]
        if allprob:
            cols.extend(["in_frame_all_prob"])
                         
        header = " ".join(sys.argv) + "\n" + "psauron score: " + str(ss)
        with open(p_out, "wt") as f:
            f.write(header + "\n")
            f.write(",".join(cols) + "\n")
            for i, desc in enumerate(description_list_clean):
                if allprob:
                    # removes all "," in data to avoid .csv errors
                    f.write(",".join([str(x).replace(",", "") for x in [desc, 
                                                                        is_protein[i],
                                                                        round(ORF_score_all[i], 5),
                                                                        ";".join([str(round(x,2)) for x in ORF_prob_allprob[i]])]]))
                    f.write("\n")
                else:
                    # removes all "," in data to avoid .csv errors
                    f.write(",".join([str(x).replace(",", "") for x in [desc, 
                                                                        is_protein[i],
                                                                        round(ORF_score_all[i], 5)]]))
                    f.write("\n")
    
    print("Done")
    
if __name__ == "__main__":
    eye_of_psauron()
    