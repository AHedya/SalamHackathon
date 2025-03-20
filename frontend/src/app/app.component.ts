import { Component, Inject, OnInit, PLATFORM_ID, afterNextRender, inject } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { initFlowbite } from 'flowbite';
import { RouterOutlet } from '@angular/router';
import { FlowbiteService } from './shared/services/flowbite.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'] 
})
export class AppComponent implements OnInit {
  title = 'Hackathon';
  flowbiteService =inject(FlowbiteService)
  constructor(){
    afterNextRender(()=>{
     initFlowbite();
    }) 
   }

  ngOnInit(): void {
    this.flowbiteService.loadFlowbite(flowbite => {
      // Your custom code here
      console.log('Flowbite loaded', flowbite);
    });
  }


  }