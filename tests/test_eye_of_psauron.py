import pytest
import unittest

from psauron.psauron import eye_of_psauron

def test_eye_of_psauron_protein():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_protein.faa', '-p']):
        eye_of_psauron()
        
def test_eye_of_psauron_CDS_allframes():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa']):
        eye_of_psauron()

def test_eye_of_psauron_CDS_singleframe():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-s']):
        eye_of_psauron()
        
def test_eye_of_psauron_CDS_CPU():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '--use-cpu']):
        eye_of_psauron()
        
def test_eye_of_psauron_CDS_verbose():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-v']):
        eye_of_psauron()
        
def test_eye_of_psauron_CDS_verbose_singleframe():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-v', '-s']):
        eye_of_psauron()

def test_eye_of_psauron_nodata():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-m', '1000000']):
        with pytest.raises(SystemExit):
            eye_of_psauron()
            
def test_eye_of_psauron_nodata_singleframe():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-m', '1000000', '-s']):
        with pytest.raises(SystemExit):
            eye_of_psauron()

def test_eye_of_psauron_exclude():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '--exclude', '1']):
        with pytest.raises(SystemExit):
            eye_of_psauron()
        
def test_eye_of_psauron_nodata_protein():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-m', '1000000', '-p']):
        with pytest.raises(SystemExit):
            eye_of_psauron()
            
def test_eye_of_psauron_exclude_protein():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '--exclude', '1', '-p']):
        with pytest.raises(SystemExit):
            eye_of_psauron()
        
def test_eye_of_psauron_badargs():
    with unittest.mock.patch('sys.argv', ['psauron']):
        with pytest.raises(SystemExit):
            eye_of_psauron()
            
def test_eye_of_psauron_badargs2():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-p', '-s']):
        with pytest.raises(SystemExit):
            eye_of_psauron()
            
def test_eye_of_psauron_allprob():
    with unittest.mock.patch('sys.argv', ['psauron', '-i', 'tests/seq_test_CDS.fa', '-p', '-s', '-a']):
        with pytest.raises(SystemExit):
            eye_of_psauron()