from engine import engine
from view import View
from VPGAgent import VPGAgent
import torch
from train_vpg import train_one_epoch_mirror 
import pandas as pd
import time
import os
import pygame

TRAINING_NAME = "training_6_mirror"
MAX_EPOCHS = 550 
RECORD_EPOCHS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550]

history = {
    "loss": [],
    "avg_return": [],
    "std_return": [], 
    "avg_len": []     
}

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
recording_path = os.path.join(root_dir, "recordings", TRAINING_NAME)
policy_path = os.path.join(root_dir, "policies", TRAINING_NAME)
analysis_path = os.path.join(root_dir, "analysis")

os.makedirs(recording_path, exist_ok=True)
os.makedirs(policy_path, exist_ok=True)
os.makedirs(analysis_path, exist_ok=True)

if __name__ == "__main__":
    my_eng = engine()
    my_agent = VPGAgent() 
    
    for param_group in my_agent.optimizer.param_groups:
        param_group['lr'] = 0.005
        
    print(f"Lancement de {TRAINING_NAME} (From Scratch + Mirroring + Anti-Jitter)...")

    starttime = time.time()
    for epoch in range (1, MAX_EPOCHS): 
        
        if epoch in RECORD_EPOCHS:
            print(f"Capture Epoch {epoch}...")
            my_view = View(my_eng)
            obs = my_eng.reset()
            done = False
            frame_count = 0
            my_eng.lvl = 2 
            
            while not done: 
                my_view.render(my_eng)
                my_view.save_frame(epoch=epoch, frame_idx=frame_count, folder=recording_path)
                
                # En démo, on utilise l'Argmax pour être propre
                with torch.no_grad():
                    logits = my_agent.policy_net(torch.as_tensor(obs, dtype=torch.float32))
                    act = torch.argmax(logits).item()
                
                obs, _, done = my_eng.step(act)
                frame_count += 1
                if frame_count > 1000: break 
            
            my_view.show_game_over(my_eng.score)
            for _ in range(10): 
                my_view.save_frame(epoch=epoch, frame_idx=frame_count, folder=recording_path)
                frame_count += 1
            
            # Sauvegarde Modèle
            torch.save(my_agent.state_dict(), os.path.join(policy_path, f"model_epoch_{epoch}.pth"))
            time.sleep(1)
            pygame.quit()
            
        loss, ret, std, length = train_one_epoch_mirror(my_agent, my_eng)

        history["loss"].append(loss)
        history["avg_return"].append(ret)
        history["std_return"].append(std)
        history["avg_len"].append(length)
        
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Retour: {ret:.1f} | Durée: {length:.0f}")
        
        if epoch % 10 == 0:
             pd.DataFrame(history).to_csv(os.path.join(analysis_path, f"{TRAINING_NAME}_progress.csv"), index=False)

    pd.DataFrame(history).to_csv(os.path.join(analysis_path, f"{TRAINING_NAME}_results.csv"), index=False)
    torch.save(my_agent.state_dict(), os.path.join(policy_path, "model_final.pth"))
    print("✅ Training 6 terminé.")
    endtime = time.time()
    dt = int(endtime - starttime)
    hours = dt // 3600
    minutes = (dt%3600) //60
    sec = (dt%3600)%60
    print(f"Training time : {hours}h {minutes}min {sec}s")