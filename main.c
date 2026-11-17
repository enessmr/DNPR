#define _POSIX_C_SOURCE 200809L
#include <SDL.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <xf86drmMode.h>
#ifndef DNPR_ASSET_BUILDER
#include "panic_sim.h"
#endif

static int screen_w=1920,screen_h=1080;
#define W screen_w
#define H screen_h
#define PI 3.14159265358979323846f
#define MAX_VERT 4096
#define MAX_TRI 8192

typedef struct { float x,y,z; } Vec3;
typedef struct { Vec3 p,n; uint8_t r,g,b; } Vertex;
typedef struct { uint16_t a,b,c; uint8_t r,g,bcolor; } Tri;
typedef struct { Vertex v[MAX_VERT]; Tri t[MAX_TRI]; int nv,nt; } Mesh;
typedef struct { float x,y,z,invz,lr,lg,lb,nx,ny,nz; } Projected;

static Mesh mesh;
static uint32_t *pixels,*background,*final_head_cache;
static struct drm_buffer {
    int fd;
    uint32_t handle,pitch;
    size_t size;
    void *map;
} drm_buffers[2]={{.fd=-1},{.fd=-1}};
static float *depth;
static pthread_t raster_workers[15];
static pthread_mutex_t raster_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t raster_start=PTHREAD_COND_INITIALIZER;
static pthread_cond_t raster_done=PTHREAD_COND_INITIALIZER;
static const Projected *raster_vertices;
static atomic_int raster_next_band;
static int raster_band_count;
static unsigned raster_generation,raster_completed;
static bool raster_stop;
static int raster_worker_count;
static uint8_t *background_rows,*star_images,*prompt_images,*prompt_images_hd,*sparkle_images,*sparkle_images_hd;
static volatile sig_atomic_t running=1;
static int server_fd=-1;
static char socket_path[sizeof(((struct sockaddr_un *)0)->sun_path)];

static Vec3 v3(float x,float y,float z){ return (Vec3){x,y,z}; }
static Vec3 add(Vec3 a,Vec3 b){return v3(a.x+b.x,a.y+b.y,a.z+b.z);}
static Vec3 mul(Vec3 a,float s){return v3(a.x*s,a.y*s,a.z*s);}
static float dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static Vec3 norm(Vec3 a){float d=sqrtf(dot(a,a));return d>0?mul(a,1/d):v3(0,0,1);}
static uint32_t rgb(int r,int g,int b){return 0xff000000u|((uint32_t)r<<16)|((uint32_t)g<<8)|(uint32_t)b;}
static int clampi(int v){return v<0?0:v>255?255:v;}

static Vec3 head_frames[131][MAX_VERT];
static Vec3 star_positions[131][2];
static int head_frame=-1,head_next=-1;
static float head_frac=-1;
static int selected_frame=-1;
static int selected_time_ms=-1;

static bool load_head(void){
    char path[4096];
    ssize_t len=readlink("/proc/self/exe",path,sizeof(path)-24);
    if(len<=0)return false;
    path[len]=0;
    char *slash=strrchr(path,'/');if(!slash)return false;
    strcpy(slash+1,"mario_head.bin");
    FILE *f=fopen(path,"rb");if(!f){perror(path);return false;}
    char magic[8];uint32_t counts[3];
    if(fread(magic,1,8,f)!=8||memcmp(magic,"DNPRM001",8)||fread(counts,4,3,f)!=3||
        counts[0]>MAX_VERT||counts[1]>MAX_TRI||counts[2]!=131){
        fputs("Invalid baked head asset\n",stderr);fclose(f);return false;
    }
    mesh.nv=(int)counts[0];mesh.nt=(int)counts[1];
    for(int i=0;i<mesh.nt;i++){
        unsigned char raw[9];if(fread(raw,1,9,f)!=9){fclose(f);return false;}
        Tri *t=&mesh.t[i];
        t->a=(uint16_t)(raw[0]|raw[1]<<8);
        t->b=(uint16_t)(raw[2]|raw[3]<<8);
        t->c=(uint16_t)(raw[4]|raw[5]<<8);
        t->r=raw[6];t->g=raw[7];t->bcolor=raw[8];
        if(t->a>=mesh.nv||t->b>=mesh.nv||t->c>=mesh.nv){fclose(f);return false;}
    }
    for(int frame=0;frame<131;frame++){
        for(int i=0;i<mesh.nv;i++){
            float xyz[3];if(fread(xyz,sizeof(float),3,f)!=3){fclose(f);return false;}
            head_frames[frame][i]=v3(xyz[0],xyz[1],xyz[2]);
        }
        for(int i=0;i<2;i++){
            float xyz[3];if(fread(xyz,sizeof(float),3,f)!=3){fclose(f);return false;}
            star_positions[frame][i]=v3(xyz[0],xyz[1],xyz[2]);
        }
    }
    fclose(f);return true;
}

static bool load_scene(void){
    char path[4096];ssize_t len=readlink("/proc/self/exe",path,sizeof(path)-24);
    if(len<=0)return false;path[len]=0;
    char *slash=strrchr(path,'/');if(!slash)return false;
    strcpy(slash+1,"scene.bin");
    FILE *f=fopen(path,"rb");if(!f){perror(path);return false;}
    char magic[8];uint32_t size[4];
    if(fread(magic,1,8,f)!=8||memcmp(magic,"DNPRS005",8)||fread(size,4,4,f)!=4||
        size[0]!=1200||size[1]!=300||size[2]!=128||size[3]!=128){
        fputs("Invalid scene asset\n",stderr);fclose(f);return false;
    }
    size_t bg_size=4u*1200*300*3,star_size=16u*128*128*4,prompt_size=11u*16*16*4;
    size_t prompt_hd_size=11u*128*128*4,sparkle_size=6u*32*32*4;
    size_t sparkle_hd_size=6u*128*128*4;
    background_rows=malloc(bg_size);star_images=malloc(star_size);
    prompt_images=malloc(prompt_size);prompt_images_hd=malloc(prompt_hd_size);
    sparkle_images=malloc(sparkle_size);sparkle_images_hd=malloc(sparkle_hd_size);
    if(!background_rows||!star_images||!prompt_images||!prompt_images_hd||
       !sparkle_images||!sparkle_images_hd){fclose(f);return false;}
    bool ok=fread(background_rows,1,bg_size,f)==bg_size&&
        fread(star_images,1,star_size,f)==star_size&&
        fread(prompt_images,1,prompt_size,f)==prompt_size&&
        fread(prompt_images_hd,1,prompt_hd_size,f)==prompt_hd_size&&
        fread(sparkle_images,1,sparkle_size,f)==sparkle_size&&
        fread(sparkle_images_hd,1,sparkle_hd_size,f)==sparkle_hd_size;
    fclose(f);return ok;
}

static bool init_buffers(int width,int height){
    if(width<640||height<360||width>4096||height>2160){
        fputs("Unsupported display dimensions\n",stderr);return false;
    }
    screen_w=width;screen_h=height;
    size_t count=(size_t)W*H;
    pixels=malloc(count*sizeof(*pixels));
    background=malloc(count*sizeof(*background));
    final_head_cache=malloc(count*sizeof(*final_head_cache));
    depth=malloc(count*sizeof(*depth));
    if(!pixels||!background||!final_head_cache||!depth)return false;
    // Keep the 16:9 tile proportions on wider displays and repeat to fill.
    int tile_width=(int64_t)W*9>(int64_t)H*16?H*4/9:W/4;
    for(int y=0;y<H;y++){
        int row=(int)(((int64_t)y*12/H)%4);
        int sy=(int)(((int64_t)y*12*300/H)%300);
        for(int x=0;x<W;x++){
            int sx=(int)(((int64_t)x*1200/tile_width)%1200);
            size_t at=((size_t)row*1200*300+(size_t)sy*1200+sx)*3;
            background[y*W+x]=rgb(background_rows[at],background_rows[at+1],background_rows[at+2]);
        }
    }
    return true;
}

/* Allocate a CPU-mappable DRM image without becoming DRM master or changing
 * either monitor's mode. Keep a RAM source image: reads from NVIDIA's mapped
 * video memory are far too slow for SDL's window-surface presentation.
 */
static bool init_drm_buffer(struct drm_buffer *buffer,int height){
    glob_t cards={0};
    if(glob("/dev/dri/card[0-9]*",0,NULL,&cards)!=0)return false;
    for(size_t i=0;i<cards.gl_pathc;i++){
        int fd=open(cards.gl_pathv[i],O_RDWR|O_CLOEXEC);
        uint32_t handle=0,pitch=0;
        uint64_t size=0,offset=0;
        if(fd<0)continue;
        if(drmModeCreateDumbBuffer(fd,W,height,32,0,&handle,&pitch,&size)!=0){close(fd);continue;}
        if(pitch!=(uint32_t)W*4 || size<(uint64_t)pitch*height ||
           drmModeMapDumbBuffer(fd,handle,&offset)!=0){
            drmModeDestroyDumbBuffer(fd,handle);close(fd);continue;
        }
        void *map=mmap(NULL,(size_t)size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(off_t)offset);
        if(map==MAP_FAILED){drmModeDestroyDumbBuffer(fd,handle);close(fd);continue;}
        buffer->fd=fd;
        buffer->handle=handle;
        buffer->pitch=pitch;
        buffer->size=(size_t)size;
        buffer->map=map;
        fprintf(stderr,"DNPR: mapped DRM buffer on %s\n",cards.gl_pathv[i]);
        globfree(&cards);
        return true;
    }
    globfree(&cards);
    return false;
}

static void release_drm_buffer(struct drm_buffer *buffer){
    if(!buffer->map)return;
    munmap(buffer->map,buffer->size);
    drmModeDestroyDumbBuffer(buffer->fd,buffer->handle);
    close(buffer->fd);
    buffer->map=NULL;
    buffer->fd=-1;
}
#ifndef DNPR_ASSET_BUILDER
static bool init_panic_scanouts(void){
    for(unsigned int index=0;index<2;index++){
        if(!init_drm_buffer(&drm_buffers[index],1152)||
           drm_buffers[index].pitch!=(uint32_t)W*4||
           drm_buffers[index].size<(size_t)W*4*1152){
            for(unsigned int buffer=0;buffer<2;buffer++)release_drm_buffer(&drm_buffers[buffer]);
            return false;
        }
    }
    for(unsigned int index=0;index<2;index++)
        if(!panic_sim_set_scanout(index,drm_buffers[index].map,
                                  drm_buffers[index].size,drm_buffers[index].pitch))abort();
    return true;
}
#endif

static void pose_head(int frame,int next,float frac){
    if(next>130)next=130;
    if(next<frame)next=frame;
    if(frame==head_frame&&next==head_next&&frac==head_frac)return;
    head_frame=frame;head_next=next;head_frac=frac;
    for(int i=0;i<mesh.nv;i++){
        Vec3 a=head_frames[frame][i],b=head_frames[next][i];
        mesh.v[i].p=v3(a.x+(b.x-a.x)*frac,a.y+(b.y-a.y)*frac,a.z+(b.z-a.z)*frac);
        mesh.v[i].n=v3(0,0,0);
        mesh.v[i].r=mesh.v[i].g=mesh.v[i].b=255;
    }
    for(int i=0;i<mesh.nt;i++){
        Tri t=mesh.t[i];
        Vec3 a=mesh.v[t.a].p,b=mesh.v[t.b].p,c=mesh.v[t.c].p;
        Vec3 u=v3(b.x-a.x,b.y-a.y,b.z-a.z),v=v3(c.x-a.x,c.y-a.y,c.z-a.z);
        Vec3 n=v3(u.z*v.y-u.y*v.z,u.x*v.z-u.z*v.x,u.y*v.x-u.x*v.y);
        mesh.v[t.a].n=add(mesh.v[t.a].n,n);
        mesh.v[t.b].n=add(mesh.v[t.b].n,n);
        mesh.v[t.c].n=add(mesh.v[t.c].n,n);
    }
    for(int i=0;i<mesh.nv;i++)mesh.v[i].n=norm(mesh.v[i].n);
}

static Projected project(Vertex v){
    Vec3 p=v.p,n=v.n;
    float z=20-p.z;
    Vec3 fixed=norm(v3(-.45,.75,1.1));
    Vec3 white=norm(v3(star_positions[head_frame][0].x-p.x,star_positions[head_frame][0].y-p.y,star_positions[head_frame][0].z-p.z));
    Vec3 red=norm(v3(star_positions[head_frame][1].x-p.x,star_positions[head_frame][1].y-p.y,star_positions[head_frame][1].z-p.z));
    float key=fmaxf(0,dot(n,fixed));
    float fill=fmaxf(0,dot(n,white));
    float warm=fmaxf(0,dot(n,red));
    float base=.42f+key*.76f+fill*.18f;
    float scale=455*(H/360.f)/z;
    return (Projected){W*.5f+p.x*scale,H*.5f-(p.y-2)*scale,z,1/z,
        base+warm*.16f,base+warm*.015f,base+warm*.015f,n.x,n.y,n.z};
}

static float edge(float ax,float ay,float bx,float by,float x,float y){return (x-ax)*(by-ay)-(y-ay)*(bx-ax);}
static void draw_tri(Projected a,Projected b,Projected c,Tri tri,int band_y0,int band_y1){
    if(a.z<=.05f||b.z<=.05f||c.z<=.05f)return;
    float area=edge(a.x,a.y,b.x,b.y,c.x,c.y);
    if(fabsf(area)<.01f)return;
    float inv_area=1.f/area;
    int x0=(int)fmaxf(0,floorf(fminf(a.x,fminf(b.x,c.x))));
    int x1=(int)fminf(W-1,ceilf(fmaxf(a.x,fmaxf(b.x,c.x))));
    int y0=(int)fmaxf(band_y0,floorf(fminf(a.y,fminf(b.y,c.y))));
    int y1=(int)fminf(band_y1-1,ceilf(fmaxf(a.y,fmaxf(b.y,c.y))));
    float du_dx=(c.y-b.y)*inv_area,dv_dx=(a.y-c.y)*inv_area;
    for(int y=y0;y<=y1;y++){
        float py=y+.5f;
        float row_u=edge(b.x,b.y,c.x,c.y,x0+.5f,py)*inv_area;
        float row_v=edge(c.x,c.y,a.x,a.y,x0+.5f,py)*inv_area;
    for(int x=x0;x<=x1;x++){
        int dx=x-x0;
        float u=row_u+dx*du_dx;
        float v=row_v+dx*dv_dx;
        float w=1-u-v;
        if(u<0||v<0||w<0)continue;
        float iz=u*a.invz+v*b.invz+w*c.invz;
        int idx=y*W+x;
        if(iz<=depth[idx])continue;
        depth[idx]=iz;
        float rz=1.f/iz;
        float lr=(u*a.lr*a.invz+v*b.lr*b.invz+w*c.lr*c.invz)*rz;
        float lg=(u*a.lg*a.invz+v*b.lg*b.invz+w*c.lg*c.invz)*rz;
        float lb=(u*a.lb*a.invz+v*b.lb*b.invz+w*c.lb*c.invz)*rz;
        // Interpolate normals per pixel so the broad polygons can carry a
        // highlight. Vertex-only specular loses it on the cap and nose.
        Vec3 n=norm(v3(u*a.nx+v*b.nx+w*c.nx,u*a.ny+v*b.ny+w*c.ny,u*a.nz+v*b.nz+w*c.nz));
        float h=fmaxf(0,dot(n,v3(-.46f,.32f,.83f)));
        float shine=h*h;shine*=shine;shine*=shine;shine*=shine;shine*=shine;
        shine*=shine;shine*=shine; // h^128: small, soft-edged glints
        int spec=(int)(90*shine);
        int r=(int)(tri.r*lr)+spec;
        int g=(int)(tri.g*lg)+spec;
        int bl=(int)(tri.bcolor*lb)+spec;
        pixels[idx]=rgb(clampi(r),clampi(g),clampi(bl));
    }
    }
}

#define RASTER_BAND_H 32
static void raster_band(const Projected *p,int band){
    int y0=band*RASTER_BAND_H;
    int y1=y0+RASTER_BAND_H;
    if(y1>H)y1=H;
    for(int i=0;i<mesh.nt;i++){
        Tri tri=mesh.t[i];
        draw_tri(p[tri.a],p[tri.b],p[tri.c],tri,y0,y1);
    }
}

static void raster_grab_bands(const Projected *p){
    for(;;){
        int band=atomic_fetch_add(&raster_next_band,1);
        if(band>=raster_band_count)break;
        raster_band(p,band);
    }
}

static void *raster_worker(void *arg){
    (void)arg;
    unsigned seen=0;
    pthread_mutex_lock(&raster_lock);
    for(;;){
        while(!raster_stop&&seen==raster_generation)
            pthread_cond_wait(&raster_start,&raster_lock);
        if(raster_stop)break;
        seen=raster_generation;
        const Projected *p=raster_vertices;
        pthread_mutex_unlock(&raster_lock);
        raster_grab_bands(p);
        pthread_mutex_lock(&raster_lock);
        raster_completed++;
        pthread_cond_signal(&raster_done);
    }
    pthread_mutex_unlock(&raster_lock);
    return NULL;
}

static void raster_init(void){
    long cpus=sysconf(_SC_NPROCESSORS_ONLN);
    raster_worker_count=cpus>1?(int)(cpus-1):0;
    if(raster_worker_count>15)raster_worker_count=15;
    for(int i=0;i<raster_worker_count;i++)
        if(pthread_create(&raster_workers[i],NULL,raster_worker,NULL))
            raster_worker_count=i;
}

static void raster_shutdown(void){
    pthread_mutex_lock(&raster_lock);
    raster_stop=true;
    pthread_cond_broadcast(&raster_start);
    pthread_mutex_unlock(&raster_lock);
    for(int i=0;i<raster_worker_count;i++)pthread_join(raster_workers[i],NULL);
}

static void raster_draw(const Projected *p){
    pthread_mutex_lock(&raster_lock);
    raster_vertices=p;
    raster_completed=0;
    raster_band_count=(H+RASTER_BAND_H-1)/RASTER_BAND_H;
    atomic_store(&raster_next_band,0);
    raster_generation++;
    pthread_cond_broadcast(&raster_start);
    pthread_mutex_unlock(&raster_lock);
    raster_grab_bands(p);
    pthread_mutex_lock(&raster_lock);
    while(raster_completed<(unsigned)raster_worker_count)
        pthread_cond_wait(&raster_done,&raster_lock);
    pthread_mutex_unlock(&raster_lock);
}

static void prompt_glyph(int index,int x,int y,int size){
    int source_size=H>=720?128:16;
    const uint8_t *atlas=source_size==128?prompt_images_hd:prompt_images;
    const uint8_t *src=atlas+(size_t)index*source_size*source_size*4;
    for(int dy=0;dy<size;dy++){
        int py=y+dy;if((unsigned)py>=(unsigned)H)continue;
        int sy=dy*source_size/size;
        for(int dx=0;dx<size;dx++){
            int px=x+dx;if((unsigned)px>=(unsigned)W)continue;
            const uint8_t *rgba=src+((size_t)sy*source_size+dx*source_size/size)*4;
            int a=rgba[3];if(!a)continue;
            uint32_t old=pixels[py*W+px];
            int r=(rgba[0]*a+((old>>16)&255)*(255-a))/255;
            int g=(rgba[1]*a+((old>>8)&255)*(255-a))/255;
            int b=(rgba[2]*a+(old&255)*(255-a))/255;
            pixels[py*W+px]=rgb(r,g,b);
        }
    }
}
static void draw_prompt_line(const char *message,int x,int y,int size){
    static const char *order="REBOTYUDVIC";
    int advance=size*5/6,space=size/2;
    for(const char *p=message;*p;p++){
        if(*p==' '){x+=space;continue;}
        const char *at=strchr(order,*p);
        if(at)prompt_glyph((int)(at-order),x,y,size);
        x+=advance;
    }
}
static void draw_prompt(void){
    int size=H/21,x=W*195/1000,y=H*77/100;
    draw_prompt_line("REBOOT",x,y,size);
    draw_prompt_line("YOUR",x,y+size+H/120,size);
    draw_prompt_line("DEVICE",x,y+2*(size+H/120),size);
}

static void textured_sprite_tinted(const uint8_t *source,int source_size,int cx,int cy,int size,bool red){
    int left=cx-size/2,top=cy-size/2;
    int x0=left<0?0:left,y0=top<0?0:top;
    int x1=left+size>W?W:left+size,y1=top+size>H?H:top+size;
    for(int y=y0;y<y1;y++){
        int sy=(y-top)*source_size/size;
        for(int x=x0;x<x1;x++){
            int sx=(x-left)*source_size/size;
            const uint8_t *src=source+((size_t)sy*source_size+sx)*4;
            int alpha=src[3];if(!alpha)continue;
            int sr=src[0],sg=red?src[1]/6:src[1],sb=red?src[2]/8:src[2];
            if(alpha==255){pixels[y*W+x]=rgb(sr,sg,sb);continue;}
            uint32_t dst=pixels[y*W+x];
            int r=(sr*alpha+((dst>>16)&255)*(255-alpha))/255;
            int g=(sg*alpha+((dst>>8)&255)*(255-alpha))/255;
            int b=(sb*alpha+(dst&255)*(255-alpha))/255;
            pixels[y*W+x]=rgb(r,g,b);
        }
    }
}
static void textured_sprite(const uint8_t *source,int source_size,int cx,int cy,int size){
    textured_sprite_tinted(source,source_size,cx,cy,size,false);
}

static void star_with_particles(int image,int cx,int cy,int size,unsigned sprite_tick,bool red){
    int sparkle_size=H>=720?128:32;
    const uint8_t *sparkles=H>=720?sparkle_images_hd:sparkle_images;
    if(!red){
        int sparkle_frame=(sprite_tick+4)%6;
        textured_sprite(sparkles+(size_t)sparkle_frame*sparkle_size*sparkle_size*4,sparkle_size,
            cx+size/2,cy-size/4,size*5/3);
        textured_sprite(sparkles+(size_t)((sparkle_frame+3)%6)*sparkle_size*sparkle_size*4,sparkle_size,
            cx-size/2,cy+size/4,size);
        textured_sprite(sparkles+(size_t)((sparkle_frame+1)%6)*sparkle_size*sparkle_size*4,sparkle_size,
            cx+size/2,cy-size,size/2);
    }else{
        int sparkle_frame=(sprite_tick+2)%6;
        textured_sprite_tinted(sparkles+(size_t)sparkle_frame*sparkle_size*sparkle_size*4,sparkle_size,
            cx+size,cy-size/2,size*3/2,true);
        textured_sprite_tinted(sparkles+(size_t)((sparkle_frame+3)%6)*sparkle_size*sparkle_size*4,sparkle_size,
            cx-size,cy+size/2,size,true);
        textured_sprite_tinted(sparkles+(size_t)((sparkle_frame+1)%6)*sparkle_size*sparkle_size*4,sparkle_size,
            cx+size*2,cy,size,true);
    }
    textured_sprite(star_images+(size_t)image*128*128*4,128,cx,cy,size);
}

static void frame(double seconds,bool static_frame,bool animation_only,bool overlays){
    static bool final_cached=false;
    uint64_t elapsed_ms=seconds>0?(uint64_t)(seconds*1000.0):0;
    uint64_t pose_q=elapsed_ms*30*65536/1000;
    float pose=selected_frame>=0?(float)selected_frame:(static_frame?130.f:(float)pose_q/65536.f);
    if(pose>130.f)pose=130.f;
    int frame_index=(int)pose;
    float pose_frac=pose-(float)frame_index;
    if(frame_index==130&&final_cached){
        memcpy(pixels,final_head_cache,(size_t)W*H*sizeof(*pixels));
    }else{
        if(animation_only)memset(pixels,0,(size_t)W*H*sizeof(*pixels));
        else memcpy(pixels,background,(size_t)W*H*sizeof(*pixels));
        memset(depth,0,(size_t)W*H*sizeof(*depth));
        pose_head(frame_index,frame_index<130?frame_index+1:130,pose_frac);
        Projected p[MAX_VERT];
        for(int i=0;i<mesh.nv;i++)p[i]=project(mesh.v[i]);
        raster_draw(p);
        if(frame_index==130){
            memcpy(final_head_cache,pixels,(size_t)W*H*sizeof(*pixels));
            final_cached=true;
        }
    }
    // The JSON contains one silver and one red star light. The kernel panic
    // base frame renders the same pose without overlays; the kernel composites
    // these layers live from the preloaded sprites.
    if(overlays){
        uint64_t overlay_ms=selected_frame>=0?(uint64_t)selected_frame*1000/30:
            (static_frame?130ULL*1000/30:elapsed_ms);
        unsigned sprite_tick=(unsigned)(overlay_ms*12/1000);
        int star_size=H/24;
        star_with_particles((sprite_tick+1)%8,W*21/100,H*97/100,star_size,sprite_tick,false);
        star_with_particles(8+(sprite_tick+5)%8,W*92/100,H*16/100,star_size/2,sprite_tick,true);
        if(static_frame||!((overlay_ms/500)&1)){
            draw_prompt();
        }
    }
}

static void on_signal(int sig){(void)sig;running=0;}
static int control(const char *command){
    int fd=socket(AF_UNIX,SOCK_STREAM,0);if(fd<0)return -1;
    struct sockaddr_un addr={.sun_family=AF_UNIX};
    snprintf(addr.sun_path,sizeof(addr.sun_path),"%s",socket_path);
    if(connect(fd,(struct sockaddr*)&addr,sizeof(addr))<0){close(fd);return -1;}
    if(write(fd,command,1)!=1){close(fd);return -1;}
    char reply[32]={0};ssize_t n=read(fd,reply,sizeof(reply)-1);close(fd);
    if(n>0)fputs(reply,stdout);
    return 0;
}
static void poll_control(void){
    int fd=accept(server_fd,NULL,NULL);if(fd<0)return;
    char cmd=0;if(read(fd,&cmd,1)==1){
        const char *reply=cmd=='q'?"stopping\n":"running\n";
        (void)write(fd,reply,strlen(reply));
        if(cmd=='q')running=0;
    }
    close(fd);
}
static bool create_server(void){
    server_fd=socket(AF_UNIX,SOCK_STREAM|SOCK_NONBLOCK,0);if(server_fd<0)return false;
    struct sockaddr_un addr={.sun_family=AF_UNIX};
    snprintf(addr.sun_path,sizeof(addr.sun_path),"%s",socket_path);
    if(bind(server_fd,(struct sockaddr*)&addr,sizeof(addr))<0){
        if(errno!=EADDRINUSE||control("s")==0)return false;
        unlink(socket_path);
        if(bind(server_fd,(struct sockaddr*)&addr,sizeof(addr))<0)return false;
    }
    if(listen(server_fd,4)<0)return false;
    return true;
}

static int16_t *audio_samples;
static uint32_t audio_values;
static bool load_audio(void){
    char path[4096];ssize_t len=readlink("/proc/self/exe",path,sizeof(path)-32);
    if(len<=0)return false;path[len]=0;
    char *slash=strrchr(path,'/');if(!slash)return false;
    strcpy(slash+1,"gameover_audio.bin");
    FILE *f=fopen(path,"rb");if(!f){perror(path);return false;}
    char magic[8];uint32_t header[3];
    if(fread(magic,1,8,f)!=8||memcmp(magic,"DNPA0001",8)||fread(header,4,3,f)!=3||
        header[0]!=32000||header[1]!=2||header[2]>32000*20){fclose(f);return false;}
    audio_values=header[2]*2;
    audio_samples=malloc((size_t)audio_values*sizeof(*audio_samples));
    bool ok=audio_samples&&fread(audio_samples,sizeof(*audio_samples),audio_values,f)==audio_values;
    fclose(f);return ok;
}
static void audio_callback(void *userdata,Uint8 *stream,int len){
    uint32_t *pos=userdata;int n=len/(int)sizeof(int16_t);int16_t *out=(int16_t*)stream;
    for(int i=0;i<n;i++)out[i]=*pos<audio_values?audio_samples[(*pos)++]:0;
}
static int save_ppm(const char *path){
    FILE *f=fopen(path,"wb");if(!f){perror(path);return 1;}
    fprintf(f,"P6\n%d %d\n255\n",W,H);
    for(int i=0;i<W*H;i++){uint8_t c[]={pixels[i]>>16,pixels[i]>>8,pixels[i]};fwrite(c,1,3,f);}
    fclose(f);return 0;
}
#ifndef DNPR_ASSET_BUILDER
static double monotonic_seconds(void){
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC,&now);
    return now.tv_sec+now.tv_nsec/1000000000.0;
}
static bool parse_benchmark_number(const char *value,unsigned int *number){
    char *end;
    unsigned long parsed;

    if(!value[0]||value[0]=='-')return false;
    errno=0;
    parsed=strtoul(value,&end,10);
    if(errno||*end||parsed>UINT_MAX)return false;
    *number=(unsigned int)parsed;
    return true;
}
#endif
static void usage(void){puts("dnpr-test [--no-audio] [--no-drm-buffer] [--static] [--animation] [--frame-step] [--display N]\n"
    "          [--stop] [--status] [--displays] [--benchmark-panic [--from N] [--to N] [--total-frames N]]\n"
    "          [--render-base-frame FILE] [--frame N] [--time-ms N] [--legacy-preview]\n"
    "          [--panic-threads N] (1..16, default: online CPUs, capped at 16)\n"
    "          [--unshaded | --no-specular] (simulation diagnostics only)\n"
    "With --frame-step, F advances one 30 FPS video frame and its audio slice; Escape exits.\n"
    "Preview uses a compositor-managed fullscreen window.");}

static bool acer_display(int index){
    const char *name=SDL_GetDisplayName(index);
    if(name&&(strstr(name,"Acer")||strstr(name,"ACR")))return true;
    glob_t matches={0};
    if(glob("/sys/class/drm/card*-*/edid",0,NULL,&matches)!=0)return false;
    bool found=false;
    for(size_t i=0;i<matches.gl_pathc;i++){
        const char *path=matches.gl_pathv[i];
        FILE *f=fopen(path,"rb");if(!f)continue;
        unsigned char edid[128];size_t n=fread(edid,1,sizeof(edid),f);fclose(f);
        if(n!=128)continue;
        unsigned vendor=(edid[8]<<8)|edid[9];
        bool acer=vendor==((1u<<10)|(3u<<5)|18u); /* EDID ACR */
        if(!acer)for(int j=54;j<126;j+=18){
            if(edid[j]==0&&edid[j+1]==0&&edid[j+2]==0&&edid[j+3]==0xfc)
                for(int k=j+5;k<=j+14;k++)if(!memcmp(edid+k,"Acer",4)){acer=true;break;}
            if(acer)break;
        }
        if(!acer)continue;
        const char *base=strrchr(path,'/');
        if(!base)continue;
        char connector[64]={0};
        const char *end=base;
        while(end>path&&end[-1]!='/')end--;
        if(end==path)continue;
        const char *dash=strchr(end,'-');
        if(!dash)continue;
        size_t len=(size_t)(base-dash-1);
        if(len==0||len>=sizeof(connector))continue;
        memcpy(connector,dash+1,len);
        if(name&&strstr(name,connector)){found=true;break;}
    }
    globfree(&matches);return found;
}

int main(int argc,char **argv){
    bool with_audio=true,static_frame=false,animation_only=false,displays=false,benchmark_panic=false;
    bool frame_step=false;
    bool benchmark_from_set=false,benchmark_to_set=false,benchmark_total_set=false;
    unsigned int benchmark_from=0,benchmark_to=6,benchmark_total=7;
    unsigned int panic_threads=0;
    bool drm_copy=true;
#ifndef DNPR_ASSET_BUILDER
    bool unshaded=false,no_specular=false;
#endif
#ifdef DNPR_ASSET_BUILDER
    bool panic_mode=false;
#else
    bool panic_mode=true;
#endif
    const char *render_path=NULL,*base_path=NULL;int display=-1;char command=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--audio"))with_audio=true;
        else if(!strcmp(argv[i],"--no-audio"))with_audio=false;
        else if(!strcmp(argv[i],"--static"))static_frame=true;
        else if(!strcmp(argv[i],"--animation"))animation_only=true;
        else if(!strcmp(argv[i],"--frame-step"))frame_step=true;
        else if(!strcmp(argv[i],"--displays"))displays=true;
        else if(!strcmp(argv[i],"--benchmark-panic"))benchmark_panic=true;
        else if(!strcmp(argv[i],"--panic-threads")&&i+1<argc){
#ifndef DNPR_ASSET_BUILDER
            if(!parse_benchmark_number(argv[++i],&panic_threads)||
               panic_threads<1||panic_threads>16)return 2;
#else
            return 2;
#endif
        }
        else if(!strcmp(argv[i],"--from")&&i+1<argc){
#ifndef DNPR_ASSET_BUILDER
            if(!parse_benchmark_number(argv[++i],&benchmark_from))return 2;
            benchmark_from_set=true;
#endif
        }
        else if(!strcmp(argv[i],"--to")&&i+1<argc){
#ifndef DNPR_ASSET_BUILDER
            if(!parse_benchmark_number(argv[++i],&benchmark_to))return 2;
            benchmark_to_set=true;
#endif
        }
        else if(!strcmp(argv[i],"--total-frames")&&i+1<argc){
#ifndef DNPR_ASSET_BUILDER
            if(!parse_benchmark_number(argv[++i],&benchmark_total))return 2;
            benchmark_total_set=true;
#endif
        }
        else if(!strcmp(argv[i],"--no-drm-buffer"))drm_copy=false;
        else if(!strcmp(argv[i],"--unshaded")){
#ifndef DNPR_ASSET_BUILDER
            unshaded=true;
#else
            return 2;
#endif
        }
        else if(!strcmp(argv[i],"--no-specular")){
#ifndef DNPR_ASSET_BUILDER
            no_specular=true;
#else
            return 2;
#endif
        }
        else if(!strcmp(argv[i],"--legacy-preview"))panic_mode=false;
        else if(!strcmp(argv[i],"--status"))command='s';
        else if(!strcmp(argv[i],"--stop"))command='q';
        else if(!strcmp(argv[i],"--display")&&i+1<argc)display=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--render-frame")&&i+1<argc)render_path=argv[++i];
        else if(!strcmp(argv[i],"--render-base-frame")&&i+1<argc)base_path=argv[++i];
        else if(!strcmp(argv[i],"--frame")&&i+1<argc)selected_frame=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--time-ms")&&i+1<argc)selected_time_ms=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--help")){usage();return 0;}
        else {usage();return 2;}
    }
    if(selected_frame>130||selected_frame< -1){fputs("Frame must be 0..130\n",stderr);return 2;}
    if(selected_time_ms< -1|| (selected_time_ms>=0&&selected_frame>=0)){
        fputs("Time must be nonnegative and cannot be combined with --frame\n",stderr);return 2;
    }
    if(selected_time_ms>=0&&!render_path&&!base_path){
        fputs("--time-ms requires --render-frame or --render-base-frame\n",stderr);return 2;
    }
    if(frame_step&&(benchmark_panic||render_path||base_path||static_frame||selected_frame>=0||selected_time_ms>=0)){
        fputs("--frame-step requires an animated fullscreen preview\n",stderr);return 2;
    }
#ifndef DNPR_ASSET_BUILDER
    if(unshaded&&no_specular){fputs("Choose only one shading diagnostic\n",stderr);return 2;}
    if((unshaded||no_specular)&&!panic_mode){fputs("Shading diagnostics require the panic renderer\n",stderr);return 2;}
#endif
    if(benchmark_from_set||benchmark_to_set||benchmark_total_set){
        if(!benchmark_panic){fputs("--from, --to, and --total-frames require --benchmark-panic\n",stderr);return 2;}
        if(!benchmark_total_set&&benchmark_from_set&&benchmark_to_set){
            if(benchmark_to<benchmark_from){fputs("--to must be at least --from\n",stderr);return 2;}
            benchmark_total=benchmark_to-benchmark_from+1;
        }else if(benchmark_to_set&&!benchmark_from_set){
            if(!benchmark_total||benchmark_total>benchmark_to+1ULL){
                fputs("Benchmark range starts before frame zero\n",stderr);return 2;
            }
            benchmark_from=benchmark_to-benchmark_total+1;
        }else if(!benchmark_to_set){
            if(!benchmark_total||benchmark_from+(uint64_t)benchmark_total-1>UINT_MAX){
                fputs("Invalid benchmark range\n",stderr);return 2;
            }
            benchmark_to=benchmark_from+benchmark_total-1;
        }
        if(!benchmark_total||benchmark_to<benchmark_from||
           benchmark_to-(uint64_t)benchmark_from+1!=benchmark_total||
           (uint64_t)benchmark_to*1000/30>UINT_MAX){
            fputs("Inconsistent benchmark range\n",stderr);return 2;
        }
    }
    const char *runtime=getenv("XDG_RUNTIME_DIR");if(!runtime)runtime="/tmp";
    if(snprintf(socket_path,sizeof(socket_path),"%s/dnpr-test-%u.sock",runtime,(unsigned)getuid())>=(int)sizeof(socket_path)){
        fputs("Runtime path is too long\n",stderr);return 1;
    }
    if(command){if(control(&command)<0){puts("not running");return command=='s'?0:1;}return 0;}
    if(panic_threads){
#ifndef DNPR_ASSET_BUILDER
        if(!panic_mode){fputs("--panic-threads requires the panic renderer\n",stderr);return 2;}
        panic_sim_set_threads(panic_threads);
#endif
    }
    if(benchmark_panic){
#ifndef DNPR_ASSET_BUILDER
        W=1920;H=1080;
        pixels=malloc((size_t)W*H*sizeof(*pixels));
        if(!pixels||!panic_sim_init())return 1;
        printf("panic raster threads: %u\n",panic_sim_thread_count());
        panic_sim_set_unshaded(unshaded);
        panic_sim_set_no_specular(no_specular);
        if(drm_copy&&!init_panic_scanouts())
            fputs("DNPR: mapped scanouts unavailable; benchmarking RAM fallback\n",stderr);
        if(benchmark_from)
            panic_sim_render((unsigned int)((uint64_t)(benchmark_from-1)*1000/30),
                             benchmark_from-1,pixels);
        double total=0;
        double animated_total=0,settled_total=0;
        unsigned int animated_count=0,settled_count=0;
        struct panic_sim_timing stages={0};
        for(unsigned int index=benchmark_from;;index++){
            struct panic_sim_timing timing;
            double begin=monotonic_seconds();
            panic_sim_render_timed((unsigned int)((uint64_t)index*1000/30), index,
                                   pixels,&timing);
            double elapsed=monotonic_seconds()-begin;
            total+=elapsed;
            if(timing.head_changed){animated_total+=elapsed;animated_count++;}
            else {settled_total+=elapsed;settled_count++;}
            stages.prepare_ms+=timing.prepare_ms;stages.raster_ms+=timing.raster_ms;
            stages.compose_ms+=timing.compose_ms;stages.blit_ms+=timing.blit_ms;
            stages.copy_ms+=timing.copy_ms;
            printf("frame %u: %.2f ms (raster %.2f, compose %.2f, blit %.2f)\n",
                   index,elapsed*1000,timing.raster_ms,timing.compose_ms,timing.blit_ms);
            if(index==benchmark_to)break;
        }
        printf("%u frames (%u..%u): %.2f ms average, %.2f fps render throughput\n",
               benchmark_total,benchmark_from,benchmark_to,
               total*1000/benchmark_total,benchmark_total/total);
        if(animated_count)
            printf("animated: %u frames, %.2f ms average, %.2f fps render throughput\n",
                   animated_count,animated_total*1000/animated_count,
                   animated_count/animated_total);
        if(settled_count)
            printf("settled: %u frames, %.2f ms average, %.2f fps render throughput\n",
                   settled_count,settled_total*1000/settled_count,
                   settled_count/settled_total);
        printf("average stages: prepare %.2f, raster %.2f, compose %.2f, blit %.2f, preview copy %.2f ms\n",
               stages.prepare_ms/benchmark_total,stages.raster_ms/benchmark_total,
               stages.compose_ms/benchmark_total,stages.blit_ms/benchmark_total,
               stages.copy_ms/benchmark_total);
        panic_sim_destroy();
        for(unsigned int index=0;index<2;index++)release_drm_buffer(&drm_buffers[index]);
        free(pixels);
        return 0;
#endif
    }
    if(render_path||base_path){
        if(render_path&&panic_mode){
#ifndef DNPR_ASSET_BUILDER
            W=1920;H=1080;
            pixels=malloc((size_t)W*H*sizeof(*pixels));
            if(!pixels||!panic_sim_init())return 1;
            panic_sim_set_unshaded(unshaded);
            panic_sim_set_no_specular(no_specular);
            unsigned int elapsed_ms=selected_time_ms>=0?(unsigned int)selected_time_ms:
                selected_frame>=0?(unsigned int)selected_frame*1000/30:130U*1000/30;
            panic_sim_render(elapsed_ms,0,pixels);
            int result=save_ppm(render_path);
            panic_sim_destroy();free(pixels);
            return result;
#endif
        }
        if(!load_head()||!load_scene()||!init_buffers(1920,1080))return 1;
        raster_init();
        frame(selected_time_ms>=0?selected_time_ms/1000.0:130.f/30.f,
            selected_time_ms<0,false,render_path!=NULL);
        raster_shutdown();
        return save_ppm(render_path?render_path:base_path);
    }
    if(SDL_Init(SDL_INIT_VIDEO|(with_audio?SDL_INIT_AUDIO:0))<0){fprintf(stderr,"SDL: %s\n",SDL_GetError());return 1;}
    int count=SDL_GetNumVideoDisplays();
    if(displays){for(int i=0;i<count;i++){
        SDL_Rect r;SDL_GetDisplayBounds(i,&r);
        SDL_DisplayMode mode;SDL_GetCurrentDisplayMode(i,&mode);
        printf("%d: %s %dx%d @ %d Hz %s\n",i,SDL_GetDisplayName(i),r.w,r.h,mode.refresh_rate,acer_display(i)?"(Acer target)":"");
    }SDL_Quit();return 0;}
    if(count<1){fputs("No display available\n",stderr);SDL_Quit();return 1;}
    if(display<0){for(int i=0;i<count;i++)if(acer_display(i)){display=i;break;}}
    if(display<0){fputs("Acer display not identified; refusing fullscreen preview\n",stderr);SDL_Quit();return 1;}
    if(display>=count){fputs("Invalid display index\n",stderr);SDL_Quit();return 2;}
    if(!acer_display(display)){fputs("Selected display is not identified as Acer; refusing fullscreen preview\n",stderr);SDL_Quit();return 2;}
    SDL_Rect bounds;SDL_GetDisplayBounds(display,&bounds);
    if(panic_mode){
#ifndef DNPR_ASSET_BUILDER
        if(bounds.w!=1920||bounds.h!=1080){
            fputs("Panic renderer requires a 1920x1080 display\n",stderr);SDL_Quit();return 1;
        }
        W=bounds.w;H=bounds.h;
        pixels=malloc((size_t)W*H*sizeof(*pixels));
        if(!pixels||!panic_sim_init()||(with_audio&&!load_audio())){
            panic_sim_destroy();free(pixels);SDL_Quit();return 1;
        }
        panic_sim_set_unshaded(unshaded);
        panic_sim_set_no_specular(no_specular);
        if(drm_copy&&!init_panic_scanouts())
            fputs("DNPR: mapped scanouts unavailable; using RAM fallback\n",stderr);
#endif
    }else if(!load_head()||!load_scene()||!init_buffers(bounds.w,bounds.h)||
             (with_audio&&!load_audio())){SDL_Quit();return 1;}
    if(!create_server()){fputs("Preview already running or control socket unavailable\n",stderr);SDL_Quit();return 1;}
    SDL_Window *win=SDL_CreateWindow("DNPR preview",SDL_WINDOWPOS_CENTERED_DISPLAY(display),
        SDL_WINDOWPOS_CENTERED_DISPLAY(display),bounds.w,bounds.h,
        SDL_WINDOW_BORDERLESS|SDL_WINDOW_FULLSCREEN_DESKTOP|SDL_WINDOW_ALLOW_HIGHDPI);
    if(!win){fprintf(stderr,"Window: %s\n",SDL_GetError());goto fail;}
    if(!panic_mode&&drm_copy&&!init_drm_buffer(&drm_buffers[0],H))
        fputs("DNPR: mapped DRM buffer unavailable; using the RAM image\n",stderr);
    SDL_Surface *window_surface=SDL_GetWindowSurface(win);
    SDL_Surface *image=SDL_CreateRGBSurfaceFrom(pixels,W,H,32,W*4,
        0x00ff0000u,0x0000ff00u,0x000000ffu,0xff000000u);
    if(!window_surface||!image){
        fprintf(stderr,"Window surface: %s\n",SDL_GetError());
        if(image)SDL_FreeSurface(image);
        for(unsigned int index=0;index<2;index++)release_drm_buffer(&drm_buffers[index]);
        SDL_DestroyWindow(win);goto fail;
    }
    if(!panic_mode)raster_init();
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY,"0");
    uint32_t audio_pos=0;SDL_AudioDeviceID audio=0;
    if(with_audio){SDL_AudioSpec want={0};want.freq=32000;want.format=AUDIO_S16SYS;
        want.channels=2;want.samples=1024;
        if(!frame_step){want.callback=audio_callback;want.userdata=&audio_pos;}
        audio=SDL_OpenAudioDevice(NULL,0,&want,NULL,0);
        if(audio)SDL_PauseAudioDevice(audio,0);else fprintf(stderr,"Audio: %s\n",SDL_GetError());
    }
    signal(SIGINT,on_signal);signal(SIGTERM,on_signal);
    uint64_t start=SDL_GetPerformanceCounter();double freq=(double)SDL_GetPerformanceFrequency();
    uint64_t next_frame=start+(uint64_t)(freq*6944444.0/1000000000.0);
    unsigned debug_frame=0;
    unsigned step_frame=0,step_pending=0;
    bool first_step_frame=true;
    while(running){
        SDL_Event event;while(SDL_PollEvent(&event)){
            if(event.type==SDL_QUIT||(event.type==SDL_KEYDOWN&&event.key.keysym.sym==SDLK_ESCAPE))running=0;
            if(frame_step&&event.type==SDL_KEYDOWN&&event.key.keysym.sym==SDLK_f&&
               !event.key.repeat&&step_frame+step_pending<UINT_MAX)
                step_pending++;
            if(event.type==SDL_DISPLAYEVENT&&event.display.event==SDL_DISPLAYEVENT_DISCONNECTED)running=0;
        }
        poll_control();
        if(!running)break;
        if(frame_step&&!first_step_frame&&!step_pending){SDL_Delay(10);continue;}
        if(frame_step){
            if(first_step_frame){
                first_step_frame=false;
                puts("frame 0 (paused; press F to advance)");
            }
            else{
                step_frame++;
                step_pending--;
                if(audio){
                    uint64_t begin=(uint64_t)step_frame*32000/30;
                    uint64_t end=(uint64_t)(step_frame+1)*32000/30;
                    SDL_ClearQueuedAudio(audio);
                    if(begin<audio_values/2){
                        if(end>audio_values/2)end=audio_values/2;
                        if(SDL_QueueAudio(audio,audio_samples+begin*2,
                                          (Uint32)((end-begin)*2*sizeof(*audio_samples)))<0)
                            fprintf(stderr,"Audio queue: %s\n",SDL_GetError());
                    }
                }
                printf("frame %u\n",step_frame);
                fflush(stdout);
            }
        }
        uint64_t now=SDL_GetPerformanceCounter();
        if(panic_mode&&!frame_step&&now<next_frame){
            Uint32 delay=(Uint32)((next_frame-now)*1000/freq);
            if(delay)SDL_Delay(delay);
            while((now=SDL_GetPerformanceCounter())<next_frame){}
        }
        now=SDL_GetPerformanceCounter();
        double sec=frame_step?step_frame/30.0:(now-start)/freq;
        if(panic_mode&&!frame_step){
            next_frame=now+(uint64_t)(freq*6944444.0/1000000000.0);
        }
        if(panic_mode){
#ifndef DNPR_ASSET_BUILDER
            panic_sim_render(static_frame?130U*1000/30:
                             frame_step?(unsigned int)((uint64_t)step_frame*1000/30):(unsigned int)(sec*1000),
                             ++debug_frame,pixels);
#endif
        }else{
            frame(sec,static_frame,animation_only,true);
            unsigned bar_w=((++debug_frame%40)+1)*8;
            if(bar_w>(unsigned)W)bar_w=(unsigned)W;
            for(unsigned y=0;y<4&&y<(unsigned)H;y++)
                for(unsigned x=0;x<bar_w;x++)pixels[(size_t)y*W+x]=rgb(255,0,255);
        }
        if(!panic_mode&&drm_buffers[0].map)
            memcpy(drm_buffers[0].map,pixels,(size_t)W*H*sizeof(*pixels));
        /* Fullscreen settling can invalidate SDL's window surface. */
        window_surface=SDL_GetWindowSurface(win);
        if(!window_surface||SDL_BlitScaled(image,NULL,window_surface,NULL)!=0||
           SDL_UpdateWindowSurface(win)!=0){
            fprintf(stderr,"Window present: %s\n",SDL_GetError());break;
        }
        if(!panic_mode&&!frame_step)SDL_Delay(1);
    }
    if(audio)SDL_CloseAudioDevice(audio);
    if(!panic_mode)raster_shutdown();
    SDL_FreeSurface(image);SDL_DestroyWindow(win);
fail:
    close(server_fd);unlink(socket_path);SDL_Quit();
    if(panic_mode){
#ifndef DNPR_ASSET_BUILDER
        panic_sim_destroy();
#endif
    }
    for(unsigned int index=0;index<2;index++)release_drm_buffer(&drm_buffers[index]);
    free(pixels);free(background);free(final_head_cache);free(depth);
    free(background_rows);free(star_images);free(prompt_images);free(prompt_images_hd);
    free(sparkle_images);free(sparkle_images_hd);free(audio_samples);
    return 0;
}
