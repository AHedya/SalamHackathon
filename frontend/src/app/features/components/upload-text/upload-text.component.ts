import { Component, Output,EventEmitter } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { UploadSoundService } from '../../../shared/services/upload-sound/upload-sound.service';

@Component({
  selector: 'app-upload-text',
  imports: [FormsModule],
  templateUrl: './upload-text.component.html',
  styleUrl: './upload-text.component.css'
})
export class UploadTextComponent {
  
  badges: { text: string }[] = [];
  paragraphs: { text: string }[] = [];
  isPopupOpen = false;
  editorText: string = '';
  isDrawerOpen = false;
  isAudio = false;
  isModalDrag = false;
  selectedParagraph: string | null = null;
  audio: HTMLAudioElement | null = null;

  togglePump() {
    this.isDrawerOpen = !this.isDrawerOpen;
  }

  dragAndDrop(){
    this.isModalDrag = !this.isModalDrag;
  }
  onPublish(event: Event): void {
    event.preventDefault();

    if (!this.editorText.trim()) return;

    // Split text into words
    const words = this.editorText.trim().split(/\s+/);

    // Create paragraphs of 50 words each
    this.paragraphs = [];
    for (let i = 0; i < words.length; i += 100) {
      this.paragraphs.push({ text: words.slice(i, i + 100).join(' ') });
    }
    console.log('Emitting paragraphs:', this.paragraphs);
    // Clear the textarea
    this.editorText = '';
    this.isDrawerOpen = false;
  }

  onTextSelect() {
    const selectedText = window.getSelection()?.toString().trim();
    if (selectedText && !this.badges.some((b) => b.text === selectedText)) {
      const badge = { text: selectedText };
      this.badges.push(badge);
      setTimeout(
        () => (this.badges = this.badges.filter((b) => b !== badge)),
        10000
      );
    }
  }
  // Handle badge click to "send" the selected text
  sendBadgeText(badgeText: string) {
    console.log(`Sending selected text: ${badgeText}`);
    // Add your logic here to send the text (e.g., to an API or elsewhere)
  }

  logParagraph(paragraph: string): void {
    console.log(paragraph);
    this.selectedParagraph = paragraph;
    this.isAudio = !this.isAudio;
  }

  constructor(private uploadSound: UploadSoundService) {}

  readText(text: string) {
    if (!text.trim()) {
      console.log('لا يوجد نص للقراءة!');
      return;
    }

    this.uploadSound.getAudio(text).subscribe(
      (audioBlob) => {
        if (this.audio) {
          this.audio.pause();
        }
        const audioUrl = URL.createObjectURL(audioBlob);
        this.audio = new Audio(audioUrl);
        this.audio.play();
      },
      (error) => {
        console.error('خطأ أثناء استدعاء API الصوت:', error);
        alert('تعذر تشغيل الصوت، تحقق من الاتصال بالـ API.');
      }
    );
  }

  stopReading() {
    if (this.audio) {
      this.audio.pause();
      this.audio = null;
    }
  }

}

