import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-chat',
  imports: [FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css'
})
export class ChatComponent {
  answerText: string = '';
  questionText: string = ''; 

  speakAnswer() {
    if (this.answerText) {
      const speech = new SpeechSynthesisUtterance(this.answerText);
      speech.lang = 'ar-SA'; 
      speech.volume = 1; 
      speech.rate = 1;
      speech.pitch = 1; 
      window.speechSynthesis.speak(speech);
    } else {
      alert('لا توجد إجابة للاستماع إليها!');
    }
  }


  submitQuestion() {
    this.answerText = this.questionText;
  }

}
