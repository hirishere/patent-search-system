import json
import re
import os
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import glob
import time
from collections import defaultdict
import threading

class SearchType(Enum):
    """Enum for different types of searches"""
    KEYWORD = "keyword"
    PATENT_ID = "patent_id"
    CLAIM_SEARCH = "claim_search"
    CLASSIFICATION = "classification"
    DATE_RANGE = "date_range"
    ABSTRACT_SEARCH = "abstract_search"
    TITLE_SEARCH = "title_search"
    HYBRID = "hybrid"

@dataclass
class Patent:
    """Data class to represent a patent"""
    title: Optional[str] = None
    doc_number: Optional[str] = None
    filename: Optional[str] = None
    abstract: Optional[str] = None
    detailed_description: Optional[List[str]] = None
    claims: Optional[List[str]] = None
    bibtex: Optional[str] = None
    classification: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert patent to dictionary"""
        return {
            'title': self.title,
            'doc_number': self.doc_number,
            'filename': self.filename,
            'abstract': self.abstract,
            'detailed_description': self.detailed_description,
            'claims': self.claims,
            'bibtex': self.bibtex,
            'classification': self.classification
        }

@dataclass
class HybridSearchQuery:
    """Data class for hybrid search parameters"""
    title_keywords: Optional[List[str]] = None
    abstract_keywords: Optional[List[str]] = None
    exact_title: Optional[str] = None
    classification_prefix: Optional[str] = None
    classification_exact: Optional[str] = None
    claims_keywords: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    
    def has_constraints(self) -> bool:
        """Check if any constraints are set"""
        return any([
            self.title_keywords,
            self.abstract_keywords,
            self.exact_title,
            self.classification_prefix,
            self.classification_exact,
            self.claims_keywords,
            self.date_range
        ])

class PatentScraper:
    """Class to scrape patent data from USPTO or other sources"""
    
    @staticmethod
    def scrape_uspto_patent(patent_id: str) -> Optional[Patent]:
        """Scrape patent data from USPTO website"""
        try:
            clean_id = re.sub(r'[^A-Z0-9]', '', patent_id.upper())
            url = f"https://patents.google.com/patent/{clean_id}/en"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('h1', {'class': 'patent-title'})
            abstract = soup.find('div', {'class': 'abstract'})
            claims = soup.find_all('div', {'class': 'claim'})
            
            patent = Patent(
                doc_number=clean_id,
                title=title.text.strip() if title else None,
                abstract=abstract.text.strip() if abstract else None,
                claims=[claim.text.strip() for claim in claims] if claims else None
            )
            
            return patent
            
        except Exception as e:
            return None

class InputParser:
    """Class to parse and classify user input"""
    
    @staticmethod
    def parse_input(user_input: str) -> Dict[str, Any]:
        """Parse user input and determine search type and parameters"""
        user_input = user_input.strip()
        
        # Check for hybrid search syntax
        if '{' in user_input and '}' in user_input:
            return InputParser.parse_hybrid_search(user_input)
        
        # Check for patent ID patterns
        patent_id_patterns = [
            r'^US\d{7,}[A-Z]\d?$',
            r'^\d{10,}$',
            r'^US\d{4}/\d{6,}$',
        ]
        
        for pattern in patent_id_patterns:
            if re.match(pattern, user_input, re.IGNORECASE):
                return {
                    'type': SearchType.PATENT_ID,
                    'query': user_input.upper(),
                    'params': {}
                }
        
        # Check for specific search commands
        if user_input.lower().startswith('claims:'):
            return {
                'type': SearchType.CLAIM_SEARCH,
                'query': user_input[7:].strip(),
                'params': {}
            }
        
        if user_input.lower().startswith('abstract:'):
            return {
                'type': SearchType.ABSTRACT_SEARCH,
                'query': user_input[9:].strip(),
                'params': {}
            }
        
        if user_input.lower().startswith('title:'):
            return {
                'type': SearchType.TITLE_SEARCH,
                'query': user_input[6:].strip(),
                'params': {}
            }
        
        if user_input.lower().startswith('class:'):
            return {
                'type': SearchType.CLASSIFICATION,
                'query': user_input[6:].strip(),
                'params': {}
            }
        
        # Check for date range
        date_pattern = r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})'
        date_match = re.search(date_pattern, user_input)
        if date_match:
            return {
                'type': SearchType.DATE_RANGE,
                'query': user_input,
                'params': {
                    'start_date': date_match.group(1),
                    'end_date': date_match.group(2)
                }
            }
        
        # Default to keyword search
        return {
            'type': SearchType.KEYWORD,
            'query': user_input,
            'params': {}
        }
    
    @staticmethod
    def parse_hybrid_search(user_input: str) -> Dict[str, Any]:
        """Parse hybrid search syntax"""
        try:
            start = user_input.find('{')
            end = user_input.rfind('}') + 1
            query_str = user_input[start:end]
            
            query_str = query_str.replace("'", '"')
            
            hybrid_query = HybridSearchQuery()
            
            patterns = {
                r'"?title"?\s*:\s*"([^"]+)"': 'exact_title',
                r'"?title_keywords"?\s*:\s*"([^"]+)"': 'title_keywords',
                r'"?abstract"?\s*:\s*"([^"]+)"': 'abstract_keywords',
                r'"?class_prefix"?\s*:\s*"([^"]+)"': 'classification_prefix',
                r'"?class"?\s*:\s*"([^"]+)"': 'classification_exact',
                r'"?claims"?\s*:\s*"([^"]+)"': 'claims_keywords'
            }
            
            for pattern, field in patterns.items():
                match = re.search(pattern, query_str)
                if match:
                    value = match.group(1)
                    if field.endswith('_keywords'):
                        setattr(hybrid_query, field, [k.strip() for k in value.split(',')])
                    else:
                        setattr(hybrid_query, field, value)
            
            return {
                'type': SearchType.HYBRID,
                'query': user_input,
                'params': {'hybrid_query': hybrid_query}
            }
            
        except Exception as e:
            return {
                'type': SearchType.KEYWORD,
                'query': user_input,
                'params': {}
            }

class PatentSearchEngine:
    """Main search engine class"""
    
    def __init__(self, data_directory: str = '.', handle_incomplete: str = 'include'):
        """Initialize the search engine"""
        self.data_directory = data_directory
        self.patents: List[Patent] = []
        self.incomplete_patents: List[Patent] = []
        self.parser = InputParser()
        self.scraper = PatentScraper()
        self.handle_incomplete = handle_incomplete
        self.required_fields = ['title', 'doc_number', 'abstract']
        self.stats = {
            'total_loaded': 0,
            'complete': 0,
            'incomplete': 0,
            'missing_fields': {}
        }
        
        self.indexes = {
            'classification_prefix': defaultdict(list),
            'title_words': defaultdict(set),
            'abstract_words': defaultdict(set),
        }
        
        self.load_patents()
    
    def load_patents(self):
        """Load all patents from JSON files"""
        self.patents = []
        self.incomplete_patents = []
        
        pattern = os.path.join(self.data_directory, 'patents_ipa*.json')
        files = glob.glob(pattern)
        
        print(f"Loading patents from {len(files)} files...")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for patent_data in data:
                    patent = Patent(
                        title=patent_data.get('title'),
                        doc_number=patent_data.get('doc_number'),
                        filename=patent_data.get('filename'),
                        abstract=patent_data.get('abstract'),
                        detailed_description=patent_data.get('detailed_description', []),
                        claims=patent_data.get('claims', []),
                        bibtex=patent_data.get('bibtex'),
                        classification=patent_data.get('classification')
                    )
                    
                    is_complete, missing_fields = self._check_patent_completeness(patent)
                    self.stats['total_loaded'] += 1
                    
                    if is_complete:
                        self.stats['complete'] += 1
                        self.patents.append(patent)
                    else:
                        self.stats['incomplete'] += 1
                        for field in missing_fields:
                            self.stats['missing_fields'][field] = self.stats['missing_fields'].get(field, 0) + 1
                        
                        if self.handle_incomplete == 'include':
                            self.patents.append(patent)
                        elif self.handle_incomplete == 'flag':
                            patent._incomplete = True
                            patent._missing_fields = missing_fields
                            self.patents.append(patent)
                            self.incomplete_patents.append(patent)
                    
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
        
        self._build_indexes()
        print(f"Loaded {len(self.patents)} patents successfully")
    
    def _check_patent_completeness(self, patent: Patent) -> tuple[bool, List[str]]:
        """Check if a patent has all required fields"""
        missing_fields = []
        
        for field in self.required_fields:
            value = getattr(patent, field, None)
            if value is None or (isinstance(value, (list, str)) and len(value) == 0):
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    def search(self, user_input: str, timed: bool = False) -> Dict[str, Any]:
        """Main search method that handles different types of queries"""
        start_time = time.time()
        
        parsed = self.parser.parse_input(user_input)
        search_type = parsed['type']
        query = parsed['query']
        params = parsed['params']
        
        if search_type == SearchType.PATENT_ID:
            results = self._search_by_patent_id(query)
        elif search_type == SearchType.KEYWORD:
            results = self._search_by_keyword(query)
        elif search_type == SearchType.CLAIM_SEARCH:
            results = self._search_in_claims(query)
        elif search_type == SearchType.ABSTRACT_SEARCH:
            results = self._search_in_abstracts(query)
        elif search_type == SearchType.TITLE_SEARCH:
            results = self._search_in_titles(query)
        elif search_type == SearchType.CLASSIFICATION:
            results = self._search_by_classification(query)
        elif search_type == SearchType.DATE_RANGE:
            results = self._search_by_date_range(params['start_date'], params['end_date'])
        elif search_type == SearchType.HYBRID:
            results = self._search_hybrid(params['hybrid_query'], timed=timed)
        else:
            results = {'error': 'Unknown search type'}
        
        end_time = time.time()
        
        if timed:
            results['timing'] = {
                'total_time': end_time - start_time,
                'patents_searched': len(self.patents)
            }
        
        return results
    
    def _search_by_patent_id(self, patent_id: str) -> Dict[str, Any]:
        """Search for a specific patent by ID"""
        for patent in self.patents:
            if patent.doc_number and patent_id.upper() in patent.doc_number.upper():
                return {
                    'status': 'success',
                    'source': 'local',
                    'count': 1,
                    'results': [self._format_search_result(patent)]
                }
        
        scraped_patent = self.scraper.scrape_uspto_patent(patent_id)
        
        if scraped_patent:
            return {
                'status': 'success',
                'source': 'scraped',
                'count': 1,
                'results': [self._format_search_result(scraped_patent)]
            }
        
        return {
            'status': 'success',
            'count': 0,
            'results': [],
            'message': f'Patent {patent_id} not found'
        }
    
    def _search_by_keyword(self, keyword: str) -> Dict[str, Any]:
        """Search for patents containing keyword in any field"""
        results = []
        keyword_lower = keyword.lower()
        
        for patent in self.patents:
            match_locations = []
            
            if patent.title and keyword_lower in patent.title.lower():
                match_locations.append('title')
            
            if patent.abstract and keyword_lower in patent.abstract.lower():
                match_locations.append('abstract')
            
            if patent.claims:
                for j, claim in enumerate(patent.claims):
                    if keyword_lower in claim.lower():
                        match_locations.append(f'claim_{j+1}')
                        break
            
            if patent.detailed_description:
                for j, paragraph in enumerate(patent.detailed_description):
                    if keyword_lower in paragraph.lower():
                        match_locations.append(f'description_para_{j+1}')
                        break
            
            if match_locations:
                result = self._format_search_result(patent)
                result['match_locations'] = match_locations
                results.append(result)
        
        return {
            'status': 'success',
            'query': keyword,
            'count': len(results),
            'results': results[:10]
        }
    
    def _search_in_claims(self, query: str) -> Dict[str, Any]:
        """Search specifically in patent claims"""
        results = []
        query_lower = query.lower()
        
        for patent in self.patents:
            if patent.claims:
                matching_claims = []
                for i, claim in enumerate(patent.claims):
                    if query_lower in claim.lower():
                        matching_claims.append({
                            'claim_number': i + 1,
                            'claim_text': claim[:200] + '...' if len(claim) > 200 else claim
                        })
                
                if matching_claims:
                    result = self._format_search_result(patent)
                    result['matching_claims'] = matching_claims
                    results.append(result)
        
        return {
            'status': 'success',
            'query': query,
            'search_type': 'claims',
            'count': len(results),
            'results': results[:10]
        }
    
    def _search_in_abstracts(self, query: str) -> Dict[str, Any]:
        """Search specifically in patent abstracts"""
        results = []
        query_lower = query.lower()
        
        for patent in self.patents:
            if patent.abstract and query_lower in patent.abstract.lower():
                result = self._format_search_result(patent)
                result['abstract_snippet'] = self._get_snippet(patent.abstract, query)
                results.append(result)
        
        return {
            'status': 'success',
            'query': query,
            'search_type': 'abstracts',
            'count': len(results),
            'results': results[:10]
        }
    
    def _search_in_titles(self, query: str) -> Dict[str, Any]:
        """Search specifically in patent titles"""
        results = []
        query_lower = query.lower()
        
        for patent in self.patents:
            if patent.title and query_lower in patent.title.lower():
                results.append(self._format_search_result(patent))
        
        return {
            'status': 'success',
            'query': query,
            'search_type': 'titles',
            'count': len(results),
            'results': results[:10]
        }
    
    def _search_by_classification(self, classification: str) -> Dict[str, Any]:
        """Search by patent classification"""
        results = []
        class_upper = classification.upper()
        
        for patent in self.patents:
            if patent.classification and class_upper in patent.classification.upper():
                results.append(self._format_search_result(patent))
        
        return {
            'status': 'success',
            'classification': classification,
            'count': len(results),
            'results': results[:10]
        }
    
    def _search_by_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Search by date range based on filename"""
        results = []
        
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            for patent in self.patents:
                if patent.filename:
                    date_match = re.search(r'-(\d{8})\.', patent.filename)
                    if date_match:
                        file_date = datetime.strptime(date_match.group(1), '%Y%m%d')
                        if start <= file_date <= end:
                            results.append(self._format_search_result(patent))
            
            return {
                'status': 'success',
                'start_date': start_date,
                'end_date': end_date,
                'count': len(results),
                'results': results[:10]
            }
            
        except ValueError as e:
            return {
                'status': 'error',
                'message': f'Invalid date format: {str(e)}'
            }
    
    def _search_hybrid(self, hybrid_query: HybridSearchQuery, timed: bool = False) -> Dict[str, Any]:
        """Perform hybrid search with multiple constraints"""
        start_time = time.time()
        
        candidates = set(range(len(self.patents)))
        
        if hybrid_query.classification_prefix:
            prefix_candidates = set()
            prefix = hybrid_query.classification_prefix.upper()
            for idx in candidates:
                patent = self.patents[idx]
                if patent.classification and patent.classification.upper().startswith(prefix):
                    prefix_candidates.add(idx)
            candidates &= prefix_candidates
        
        if hybrid_query.title_keywords:
            title_candidates = set()
            for keyword in hybrid_query.title_keywords:
                keyword_lower = keyword.lower()
                for idx in candidates:
                    patent = self.patents[idx]
                    if patent.title and keyword_lower in patent.title.lower():
                        title_candidates.add(idx)
            candidates &= title_candidates
        
        if hybrid_query.abstract_keywords:
            abstract_candidates = set()
            for keyword in hybrid_query.abstract_keywords:
                keyword_lower = keyword.lower()
                for idx in candidates:
                    patent = self.patents[idx]
                    if patent.abstract and keyword_lower in patent.abstract.lower():
                        abstract_candidates.add(idx)
            candidates &= abstract_candidates
        
        if hybrid_query.claims_keywords:
            claims_candidates = set()
            for keyword in hybrid_query.claims_keywords:
                keyword_lower = keyword.lower()
                for idx in candidates:
                    patent = self.patents[idx]
                    if patent.claims:
                        for claim in patent.claims:
                            if keyword_lower in claim.lower():
                                claims_candidates.add(idx)
                                break
            candidates &= claims_candidates
        
        results = []
        for idx in list(candidates)[:10]:
            patent = self.patents[idx]
            result = self._format_search_result(patent)
            results.append(result)
        
        end_time = time.time()
        
        response = {
            'status': 'success',
            'search_type': 'hybrid',
            'count': len(results),
            'results': results,
            'constraints_applied': {
                'classification_prefix': hybrid_query.classification_prefix is not None,
                'title_keywords': hybrid_query.title_keywords is not None,
                'abstract_keywords': hybrid_query.abstract_keywords is not None,
                'claims_keywords': hybrid_query.claims_keywords is not None
            }
        }
        
        if timed:
            response['timing'] = {
                'search_time': end_time - start_time,
                'total_candidates': len(self.patents),
                'filtered_candidates': len(candidates)
            }
        
        return response
    
    def _build_indexes(self):
        """Build indexes for faster searching"""
        for idx, patent in enumerate(self.patents):
            if patent.classification:
                prefix_match = re.match(r'^([A-Z]\d{2}[A-Z])', patent.classification)
                if prefix_match:
                    prefix = prefix_match.group(1)
                    self.indexes['classification_prefix'][prefix].append(idx)
            
            if patent.title:
                words = self._tokenize(patent.title)
                for word in words:
                    self.indexes['title_words'][word].add(idx)
            
            if patent.abstract:
                words = self._tokenize(patent.abstract)
                for word in words:
                    self.indexes['abstract_words'][word].add(idx)
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into lowercase words"""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return set(words)
    
    def _format_search_result(self, patent: Patent) -> Dict[str, Any]:
        """Format a patent for search results"""
        result = {
            'title': patent.title or 'Untitled',
            'doc_number': patent.doc_number or 'No document number',
            'abstract': patent.abstract or 'No abstract available',
            'classification': patent.classification or 'N/A',
            'available_fields': {
                'has_claims': bool(patent.claims),
                'has_description': bool(patent.detailed_description),
                'has_abstract': bool(patent.abstract)
            }
        }
        
        if hasattr(patent, '_incomplete') and patent._incomplete:
            result['data_quality'] = {
                'incomplete': True,
                'missing_fields': getattr(patent, '_missing_fields', [])
            }
        
        return result
    
    def _get_snippet(self, text: str, query: str, context_length: int = 150) -> str:
        """Get a text snippet around the query match"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        pos = text_lower.find(query_lower)
        if pos == -1:
            return text[:context_length] + '...'
        
        start = max(0, pos - context_length // 2)
        end = min(len(text), pos + len(query) + context_length // 2)
        
        snippet = text[start:end]
        if start > 0:
            snippet = '...' + snippet
        if end < len(text):
            snippet = snippet + '...'
        
        return snippet
    
    def get_data_completeness_report(self) -> Dict[str, Any]:
        """Get a detailed report on data completeness"""
        return {
            'total_patents': self.stats['total_loaded'],
            'complete_patents': self.stats['complete'],
            'incomplete_patents': self.stats['incomplete'],
            'completeness_rate': f"{(self.stats['complete'] / self.stats['total_loaded'] * 100):.2f}%" if self.stats['total_loaded'] > 0 else "0%",
            'missing_fields_summary': self.stats['missing_fields'],
            'handling_mode': self.handle_incomplete
        }