from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from kafka import KafkaConsumer
import json
from datetime import datetime

# Connecting to database and creating schema
load_dotenv()
DATABASE_URI = os.getenv('DATABASE_URI')
EXTERNAL_BROKER = "3.80.138.59:9092"
CONSUMER_TOPIC = "health_events"
CONSUMER_GROUP = "health_events_consumer_group2"

engine = create_engine(DATABASE_URI, echo=False) 
Base = declarative_base()

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    Location = Column(String(150), nullable=False)
    Severity = Column(String(150), nullable=False)
    EventType = Column(String(150), nullable=False)
    Details = Column(String(150), nullable=False)
    Timestamp = Column(DateTime, nullable=False)  # Use sqlalchemy.DateTime

consumer = KafkaConsumer(
    CONSUMER_TOPIC,
    bootstrap_servers=[EXTERNAL_BROKER],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",  # Start from earliest if no committed offsets
    enable_auto_commit=True,       # Commit manually
    group_id=CONSUMER_GROUP         # Use a consistent consumer group ID
)


def update_db():
    session = Session()
    count = 0  # Counter for number of entries processed
    try:
        while True:
            # Poll the Kafka topic with a timeout
            messages = consumer.poll(timeout_ms=5000)  # Polls every 1 second
            if messages.items():
                for topic_partition, msg_list in messages.items():
                    for message in msg_list:
                        # Process the message
                        event = message.value
                        new_event = Event(
                            Location=event.get("Location"),
                            Severity=event.get("Severity"),
                            EventType=event.get("EventType"),
                            Timestamp=datetime.strptime(event.get("Timestamp"), '%Y-%m-%d %H:%M:%S'),
                            Details=event.get("Details")
                        )
                        session.add(new_event)
                        count += 1
                        session.commit()
            else:  
                print(f"Total committed entries: {count}")
                break
    except Exception as e:
        session.rollback()
        print(f"Error occurred during database update: {e}")
    finally:
        consumer.close()
        session.close()
        print("closed consumer and session")




def count_and_print_entries():
    session = Session()
    try:
        # Count the number of entries
        total_count = session.query(Event).count()
        print(f"Total number of entries in the Event table: {total_count}")

        # Query and print the first 20 entries
        events = session.query(Event).limit(20).all()
        print("\nDisplaying up to 20 entries:")
        for event in events:
            print(f"ID: {event.id}, Location: {event.Location}, Severity: {event.Severity}, "
                  f"EventType: {event.EventType}, Details: {event.Details}, Timestamp: {event.Timestamp}")
    except Exception as e:
        print(f"Error occurred while querying the database: {e}")
    finally:
        session.close()



def get_earliest_event_date():
    session = Session()
    try:
        # Query the minimum Timestamp
        earliest_date = session.query(Event).order_by(Event.Timestamp.asc()).first()
        if earliest_date:
            print(f"The earliest event date is: {earliest_date.Timestamp}")
            return earliest_date.Timestamp
        else:
            print("No events found in the database.")
            return None
    except Exception as e:
        print(f"Error occurred while querying the earliest date: {e}")
        return None
    finally:
        session.close()






if __name__ == '__main__':  
    #Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)  # Create tables with updated schema
    Session = sessionmaker(bind=engine)
    update_db()
    #count_and_print_entries()


