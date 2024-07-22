from psauron.psauron import reverse_complement

def test_reverse_complement_default():
    revcomp = reverse_complement("GATACA")
    assert revcomp == "TGTATC"
    
def test_reverse_complement_N():
    revcomp = reverse_complement("NNNNNNA")
    assert revcomp == "TAAAAAA"