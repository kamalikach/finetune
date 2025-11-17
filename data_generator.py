import json
import random
from typing import List, Dict, Any

class JSONDatasetGenerator:
    """Generate synthetic training data for JSON output fine-tuning"""
    
    def __init__(self):
        self.names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        self.cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix", "Philadelphia"]
        self.products = ["laptop", "phone", "tablet", "monitor", "keyboard", "mouse"]
        self.colors = ["red", "blue", "green", "yellow", "black", "white"]
        
    def generate_simple_extraction(self, n: int = 100) -> List[Dict]:
        """Generate simple key-value extraction tasks"""
        examples = []
        for _ in range(n):
            name = random.choice(self.names)
            age = random.randint(18, 80)
            city = random.choice(self.cities)
            
            prompt = f"Extract the information: {name} is {age} years old and lives in {city}."
            output = {"name": name, "age": age, "city": city}
            
            examples.append({
                "instruction": "Extract the information as JSON.",
                "input": prompt,
                "output": json.dumps(output, ensure_ascii=False)
            })
        return examples
    
    def generate_nested_objects(self, n: int = 100) -> List[Dict]:
        """Generate nested JSON structures"""
        examples = []
        for _ in range(n):
            name = random.choice(self.names)
            age = random.randint(18, 80)
            city = random.choice(self.cities)
            product = random.choice(self.products)
            price = round(random.uniform(50, 2000), 2)
            
            prompt = f"Create a purchase record: {name}, age {age}, from {city} bought a {product} for ${price}."
            output = {
                "customer": {
                    "name": name,
                    "age": age,
                    "location": city
                },
                "purchase": {
                    "item": product,
                    "price": price
                }
            }
            
            examples.append({
                "instruction": "Format as nested JSON.",
                "input": prompt,
                "output": json.dumps(output, indent=2, ensure_ascii=False)
            })
        return examples
    
    def generate_arrays(self, n: int = 100) -> List[Dict]:
        """Generate JSON with arrays"""
        examples = []
        for _ in range(n):
            name = random.choice(self.names)
            num_items = random.randint(1, 4)
            items = random.sample(self.products, num_items)
            
            prompt = f"{name} bought: {', '.join(items)}."
            output = {
                "customer": name,
                "items": items,
                "total_items": len(items)
            }
            
            examples.append({
                "instruction": "Convert to JSON with array.",
                "input": prompt,
                "output": json.dumps(output, ensure_ascii=False)
            })
        return examples
    
    def generate_with_schema(self, n: int = 100) -> List[Dict]:
        """Generate examples with explicit schemas"""
        examples = []
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            },
            "required": ["name", "age"]
        }
        
        for _ in range(n):
            name = random.choice(self.names)
            age = random.randint(18, 80)
            email = f"{name.lower()}@example.com"
            active = random.choice([True, False])
            
            prompt = f"User: {name}, Age: {age}, Email: {email}, Active: {active}"
            output = {
                "name": name,
                "age": age,
                "email": email,
                "active": active
            }
            
            examples.append({
                "instruction": f"Follow this schema and extract data:\n{json.dumps(schema, indent=2)}",
                "input": prompt,
                "output": json.dumps(output, ensure_ascii=False)
            })
        return examples
    
    def generate_edge_cases(self, n: int = 50) -> List[Dict]:
        """Generate edge cases: nulls, empty arrays, special chars"""
        examples = []
        
        # Null values
        examples.append({
            "instruction": "Extract as JSON, use null for missing values.",
            "input": "Name: Alice, Age: unknown, City: NYC",
            "output": json.dumps({"name": "Alice", "age": None, "city": "NYC"})
        })
        
        # Empty arrays
        examples.append({
            "instruction": "List purchases as JSON.",
            "input": "Bob made no purchases today.",
            "output": json.dumps({"customer": "Bob", "purchases": []})
        })
        
        # Special characters
        examples.append({
            "instruction": "Extract as JSON.",
            "input": 'Message: "Hello, World!" from user@example.com',
            "output": json.dumps({"message": "Hello, World!", "from": "user@example.com"})
        })
        
        # Numbers as strings vs integers
        examples.append({
            "instruction": "Extract as JSON with correct types.",
            "input": "ID: 12345, Name: Charlie, Score: 95.5",
            "output": json.dumps({"id": 12345, "name": "Charlie", "score": 95.5})
        })
        
        # Boolean values
        examples.append({
            "instruction": "Convert to JSON.",
            "input": "User is_active: yes, is_admin: no",
            "output": json.dumps({"is_active": True, "is_admin": False})
        })
        
        return examples
    
    def generate_error_corrections(self, n: int = 50) -> List[Dict]:
        """Generate examples that correct common JSON errors"""
        examples = []
        
        # Missing quotes
        examples.append({
            "instruction": "Fix and output valid JSON.",
            "input": "{name: Alice, age: 30}",
            "output": json.dumps({"name": "Alice", "age": 30})
        })
        
        # Trailing commas
        examples.append({
            "instruction": "Fix and output valid JSON.",
            "input": '{"name": "Bob", "age": 25,}',
            "output": json.dumps({"name": "Bob", "age": 25})
        })
        
        # Single quotes
        examples.append({
            "instruction": "Fix and output valid JSON.",
            "input": "{'name': 'Charlie'}",
            "output": json.dumps({"name": "Charlie"})
        })
        
        return examples
    
    def generate_full_dataset(self, total_examples: int = 10000) -> List[Dict]:
        """Generate a complete balanced dataset"""
        dataset = []
        
        # Distribution of example types
        dataset.extend(self.generate_simple_extraction(int(total_examples * 0.3)))
        dataset.extend(self.generate_nested_objects(int(total_examples * 0.25)))
        dataset.extend(self.generate_arrays(int(total_examples * 0.2)))
        dataset.extend(self.generate_with_schema(int(total_examples * 0.15)))
        dataset.extend(self.generate_edge_cases(int(total_examples * 0.05)))
        dataset.extend(self.generate_error_corrections(int(total_examples * 0.05)))
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "json_training_data.jsonl"):
        """Save dataset in JSONL format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"Saved {len(dataset)} examples to {filename}")
    
    def save_as_huggingface(self, dataset: List[Dict], filename: str = "json_dataset_hf.json"):
        """Save in HuggingFace datasets format"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(dataset)} examples to {filename}")


# Example usage
if __name__ == "__main__":
    generator = JSONDatasetGenerator()
    
    # Generate 1000 training examples
    dataset = generator.generate_full_dataset(total_examples=1000)
    
    # Save in JSONL format (good for streaming)
    generator.save_dataset(dataset, "json_training_data.jsonl")
    
    # Save in HuggingFace format
    generator.save_as_huggingface(dataset, "json_dataset_hf.json")
    
    # Print a sample
    print("\nSample examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {example['instruction']}")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")
