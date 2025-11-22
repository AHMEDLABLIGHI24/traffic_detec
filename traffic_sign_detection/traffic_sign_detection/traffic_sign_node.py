import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

# Le nom du Node est 'traffic_sign_detector'
class TrafficSignDetector(Node):
    
    def __init__(self):
        super().__init__("traffic_sign_detector")

        # --- 1. Paramètres ROS 2 : Configuration du Modèle et du Topic Caméra ---
        # Permet de lancer le node avec différents modèles ou topics sans recompiler.
        self.declare_parameter("model_path", "/home/ahmed/best.pt") # Chemin par défaut
        self.declare_parameter("camera_topic", "/camera/image_raw") # Topic caméra par défaut

        model_path = self.get_parameter("model_path").value
        camera_topic = self.get_parameter("camera_topic").value

        # --- 2. Chargement du Modèle YOLOv8 ---
        self.get_logger().info(f"Chargement du modèle YOLO : {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.get_logger().error(f"Erreur lors du chargement de YOLOv8 : {e}")
            # Arrêter le node si le modèle ne charge pas
            raise

        # --- 3. Définition des Classes Ciblées (Filtrage) ---
        self.target_classes = [
            'Speed_Limit_20',
            'Speed_Limit_30',
            'Stop',
            'No_overtaking',
            'Road_closed_to_all_vehicles',
            'No_entry'
        ]

        # --- 4. Traduction FR pour l'affichage ---
        self.trans = {
            'Speed_Limit_20': 'Vitesse 20',
            'Speed_Limit_30': 'Vitesse 30',
            'Stop': 'STOP',
            'No_overtaking': 'Interdit Dépasser',
            'Road_closed_to_all_vehicles': 'Interdit à tous',
            'No_entry': 'Sens Interdit'
        }

        # --- 5. Initialisation du CV Bridge ---
        self.bridge = CvBridge()

        # --- 6. Abonnements (Subscriber) et Publications (Publishers) ---
        # Abonnement à l'image RAW de la caméra
        self.create_subscription(Image, camera_topic, self.image_callback, 1) # QoS 1 pour la latence

        # Publication des panneaux détectés (Liste de labels)
        self.pub_detect = self.create_publisher(String, "/traffic_signs", 10)
        # Publication de l'image avec les boîtes dessinées
        self.pub_image = self.create_publisher(Image, "/traffic_signs/image_annotated", 10)

        self.get_logger().info(f"Traffic Sign Detector Node lancé. Abonné à {camera_topic}.")

    def image_callback(self, msg):
        """ Callback appelé à chaque nouvelle image reçue sur le topic caméra """
        
        # 1. Conversion de l'Image ROS en Image OpenCV
        try:
            # Conversion bgr8 pour OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
        except Exception as e:
            self.get_logger().error(f"Erreur conversion Image ROS vers CV : {e}")
            return

        # 2. Détection YOLOv8
        # Utilisez l'argument 'classes' de YOLOv8 pour filtrer directement les IDs si vous les connaissez
        # Sinon, l'approche manuelle (comme ci-dessous) est plus robuste avec les noms.
        results = self.model(frame, verbose=False)

        annotated = frame.copy()
        detected_labels = []

        # 3. Traitement et Filtrage des Résultats
        for result in results:
            # Extraction des données
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cid, conf in zip(boxes, class_ids, confs):

                english_label = self.model.names[cid]

                # Filtrer UNIQUEMENT les panneaux ciblés
                if english_label not in self.target_classes:
                    continue

                # Récupération du label français
                french_label = self.trans.get(english_label, english_label)
                detected_labels.append(f"{french_label} ({conf:.2f})")

                # --- 4. Dessin sur l'image Annotée ---
                # Couleur et épaisseur
                color = (0, 255, 0) # Vert
                thickness = 2
                
                # Rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Texte du Label
                cv2.putText(annotated, 
                            f"{french_label} {conf:.2f}", 
                            (x1, y1 - 5), # Position du texte au-dessus de la boîte
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 0, 0), # Couleur du fond (noir)
                            2)
                # Dessiner le rectangle de fond (amélioration visuelle)
                (w, h), _ = cv2.getTextSize(f"{french_label} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                
                cv2.putText(annotated, 
                            f"{french_label} {conf:.2f}", 
                            (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 0, 0), # Texte noir sur fond vert
                            2)

        # 5. Publier les résultats de détection
        if detected_labels:
            msg_detect = String()
            msg_detect.data = ", ".join(detected_labels)
            self.pub_detect.publish(msg_detect)

        # 6. Publier l’image annotée
        # Reconversion de l'Image OpenCV en Image ROS
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            # Assurez-vous que le timestamp de l'image est correct (important pour RViz2)
            img_msg.header.stamp = self.get_clock().now().to_msg() 
            img_msg.header.frame_id = msg.header.frame_id # Conserver le frame_id de la source
            self.pub_image.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Erreur conversion Image CV vers ROS : {e}")
            pass # Continuer sans publier l'image annotée

# --- Fonction Main ROS 2 ---
def main(args=None):
    rclpy.init(args=args)
    try:
        node = TrafficSignDetector()
        rclpy.spin(node)
    except Exception as e:
        print(f"Erreur fatale dans le node : {e}")
    finally:
        # Assurez-vous que le node est détruit même en cas d'erreur
        if 'node' in locals() and node.valid:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
