import json
import logging
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092', 
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    consumer = KafkaConsumer(
        'task_tool', 
        bootstrap_servers='localhost:9092', 
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
except KafkaError as e:
    logger.critical(f"Failed to connect to Kafka: {e}")
    exit(1)
except Exception as e:
    logger.critical(f"Unexpected error during setup: {e}", exc_info=True)
    exit(1)

logger.info(">>> Tool Worker Started.")

for msg in consumer:
    try:
        data = msg.value
        workflow_id = data.get('workflow_id', 'unknown')
        task_name = data.get('task_name', 'unknown')
        
        logger.info(f"Executing task: {task_name} for workflow: {workflow_id}")
        
        # Simple Mock Tool Logic
        if 'context' not in data:
            raise ValueError("Missing 'context' in task data")
            
        input_data = data['context']
        result = f"Analyzed data for: {input_data}. Status: VALID."
        
        producer.send('agent_ingress', {
            "status": "success",
            "workflow_id": workflow_id,
            "task_name": task_name,
            "result": result
        })
        producer.flush()  # Guaranteed delivery
        logger.info(f"Successfully processed task: {task_name}")

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        producer.send('agent_ingress', {
            "status": "failed",
            "workflow_id": workflow_id if 'workflow_id' in locals() else 'unknown',
            "task_name": task_name if 'task_name' in locals() else 'unknown',
            "error": str(e)
        })
        producer.flush()
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}", exc_info=True)
        producer.send('agent_ingress', {
            "status": "failed",
            "workflow_id": workflow_id if 'workflow_id' in locals() else 'unknown',
            "task_name": task_name if 'task_name' in locals() else 'unknown',
            "error": str(e)
        })
        producer.flush()