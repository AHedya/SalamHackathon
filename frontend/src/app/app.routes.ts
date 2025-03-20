import { Routes } from '@angular/router';

export const routes: Routes = [
    {path:'',loadComponent:()=>import("./core/pages/splash-screen/splash-screen.component").then(e=>e.SplashScreenComponent)},
    {path:'',
        loadComponent:()=>import("./core/layout/main-layout/main-layout.component").then(e=>e.MainLayoutComponent),children:[
            {path:'home',loadComponent:()=>import("./core/pages/main/home/home.component").then(e=>e.HomeComponent)},
            {path:'audio',loadComponent:()=>import("./features/pages/audio/audio.component").then(e=>e.AudioComponent)},
            {path:'books',loadComponent:()=>import("./features/pages/books/books.component").then(e=>e.BooksComponent)},
            {path:'chat',loadComponent:()=>import("./features/pages/chat/chat.component").then(e=>e.ChatComponent)},
            {path:'text',loadComponent:()=>import("./features/components/upload-text/upload-text.component").then(e=>e.UploadTextComponent)},
            {path:'setting',loadComponent:()=>import("./features/pages/setting/setting.component").then(e=>e.SettingComponent)},
        ]
    },
    {path:'',
        loadComponent:()=>import("./core/layout/auth-layout/auth-layout.component").then(e=>e.AuthLayoutComponent),children:[
            {path:'signIn',loadComponent:()=>import("./core/auth/components/sign-in/sign-in.component").then(e=>e.SignInComponent)},
            {path:'signUp',loadComponent:()=>import("./core/auth/components/sign-up/sign-up.component").then(e=>e.SignUpComponent)},      
        ]
    },
    
    {path:'welcome',loadComponent:()=>import("./core/pages/welcome/welcome.component").then(e=>e.WelcomeComponent)},

    {path:'**',loadComponent:()=>import("./core/pages/main/not-found/not-found.component").then(e=>e.NotFoundComponent)}
];
