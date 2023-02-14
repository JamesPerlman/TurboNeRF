/**
 * SPDX-License-Identifier: (WTFPL OR CC0-1.0) AND Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glad/gl.h>

#ifndef GLAD_IMPL_UTIL_C_
#define GLAD_IMPL_UTIL_C_

#ifdef _MSC_VER
#define GLAD_IMPL_UTIL_SSCANF sscanf_s
#else
#define GLAD_IMPL_UTIL_SSCANF sscanf
#endif

#endif /* GLAD_IMPL_UTIL_C_ */

#ifdef __cplusplus
extern "C" {
#endif








static void glad_gl_load_GL_VERSION_1_0(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_1_0) return;
    context->Accum = (PFNGLACCUMPROC) load(userptr, "glAccum");
    context->AlphaFunc = (PFNGLALPHAFUNCPROC) load(userptr, "glAlphaFunc");
    context->Begin = (PFNGLBEGINPROC) load(userptr, "glBegin");
    context->Bitmap = (PFNGLBITMAPPROC) load(userptr, "glBitmap");
    context->BlendFunc = (PFNGLBLENDFUNCPROC) load(userptr, "glBlendFunc");
    context->CallList = (PFNGLCALLLISTPROC) load(userptr, "glCallList");
    context->CallLists = (PFNGLCALLLISTSPROC) load(userptr, "glCallLists");
    context->Clear = (PFNGLCLEARPROC) load(userptr, "glClear");
    context->ClearAccum = (PFNGLCLEARACCUMPROC) load(userptr, "glClearAccum");
    context->ClearColor = (PFNGLCLEARCOLORPROC) load(userptr, "glClearColor");
    context->ClearDepth = (PFNGLCLEARDEPTHPROC) load(userptr, "glClearDepth");
    context->ClearIndex = (PFNGLCLEARINDEXPROC) load(userptr, "glClearIndex");
    context->ClearStencil = (PFNGLCLEARSTENCILPROC) load(userptr, "glClearStencil");
    context->ClipPlane = (PFNGLCLIPPLANEPROC) load(userptr, "glClipPlane");
    context->Color3b = (PFNGLCOLOR3BPROC) load(userptr, "glColor3b");
    context->Color3bv = (PFNGLCOLOR3BVPROC) load(userptr, "glColor3bv");
    context->Color3d = (PFNGLCOLOR3DPROC) load(userptr, "glColor3d");
    context->Color3dv = (PFNGLCOLOR3DVPROC) load(userptr, "glColor3dv");
    context->Color3f = (PFNGLCOLOR3FPROC) load(userptr, "glColor3f");
    context->Color3fv = (PFNGLCOLOR3FVPROC) load(userptr, "glColor3fv");
    context->Color3i = (PFNGLCOLOR3IPROC) load(userptr, "glColor3i");
    context->Color3iv = (PFNGLCOLOR3IVPROC) load(userptr, "glColor3iv");
    context->Color3s = (PFNGLCOLOR3SPROC) load(userptr, "glColor3s");
    context->Color3sv = (PFNGLCOLOR3SVPROC) load(userptr, "glColor3sv");
    context->Color3ub = (PFNGLCOLOR3UBPROC) load(userptr, "glColor3ub");
    context->Color3ubv = (PFNGLCOLOR3UBVPROC) load(userptr, "glColor3ubv");
    context->Color3ui = (PFNGLCOLOR3UIPROC) load(userptr, "glColor3ui");
    context->Color3uiv = (PFNGLCOLOR3UIVPROC) load(userptr, "glColor3uiv");
    context->Color3us = (PFNGLCOLOR3USPROC) load(userptr, "glColor3us");
    context->Color3usv = (PFNGLCOLOR3USVPROC) load(userptr, "glColor3usv");
    context->Color4b = (PFNGLCOLOR4BPROC) load(userptr, "glColor4b");
    context->Color4bv = (PFNGLCOLOR4BVPROC) load(userptr, "glColor4bv");
    context->Color4d = (PFNGLCOLOR4DPROC) load(userptr, "glColor4d");
    context->Color4dv = (PFNGLCOLOR4DVPROC) load(userptr, "glColor4dv");
    context->Color4f = (PFNGLCOLOR4FPROC) load(userptr, "glColor4f");
    context->Color4fv = (PFNGLCOLOR4FVPROC) load(userptr, "glColor4fv");
    context->Color4i = (PFNGLCOLOR4IPROC) load(userptr, "glColor4i");
    context->Color4iv = (PFNGLCOLOR4IVPROC) load(userptr, "glColor4iv");
    context->Color4s = (PFNGLCOLOR4SPROC) load(userptr, "glColor4s");
    context->Color4sv = (PFNGLCOLOR4SVPROC) load(userptr, "glColor4sv");
    context->Color4ub = (PFNGLCOLOR4UBPROC) load(userptr, "glColor4ub");
    context->Color4ubv = (PFNGLCOLOR4UBVPROC) load(userptr, "glColor4ubv");
    context->Color4ui = (PFNGLCOLOR4UIPROC) load(userptr, "glColor4ui");
    context->Color4uiv = (PFNGLCOLOR4UIVPROC) load(userptr, "glColor4uiv");
    context->Color4us = (PFNGLCOLOR4USPROC) load(userptr, "glColor4us");
    context->Color4usv = (PFNGLCOLOR4USVPROC) load(userptr, "glColor4usv");
    context->ColorMask = (PFNGLCOLORMASKPROC) load(userptr, "glColorMask");
    context->ColorMaterial = (PFNGLCOLORMATERIALPROC) load(userptr, "glColorMaterial");
    context->CopyPixels = (PFNGLCOPYPIXELSPROC) load(userptr, "glCopyPixels");
    context->CullFace = (PFNGLCULLFACEPROC) load(userptr, "glCullFace");
    context->DeleteLists = (PFNGLDELETELISTSPROC) load(userptr, "glDeleteLists");
    context->DepthFunc = (PFNGLDEPTHFUNCPROC) load(userptr, "glDepthFunc");
    context->DepthMask = (PFNGLDEPTHMASKPROC) load(userptr, "glDepthMask");
    context->DepthRange = (PFNGLDEPTHRANGEPROC) load(userptr, "glDepthRange");
    context->Disable = (PFNGLDISABLEPROC) load(userptr, "glDisable");
    context->DrawBuffer = (PFNGLDRAWBUFFERPROC) load(userptr, "glDrawBuffer");
    context->DrawPixels = (PFNGLDRAWPIXELSPROC) load(userptr, "glDrawPixels");
    context->EdgeFlag = (PFNGLEDGEFLAGPROC) load(userptr, "glEdgeFlag");
    context->EdgeFlagv = (PFNGLEDGEFLAGVPROC) load(userptr, "glEdgeFlagv");
    context->Enable = (PFNGLENABLEPROC) load(userptr, "glEnable");
    context->End = (PFNGLENDPROC) load(userptr, "glEnd");
    context->EndList = (PFNGLENDLISTPROC) load(userptr, "glEndList");
    context->EvalCoord1d = (PFNGLEVALCOORD1DPROC) load(userptr, "glEvalCoord1d");
    context->EvalCoord1dv = (PFNGLEVALCOORD1DVPROC) load(userptr, "glEvalCoord1dv");
    context->EvalCoord1f = (PFNGLEVALCOORD1FPROC) load(userptr, "glEvalCoord1f");
    context->EvalCoord1fv = (PFNGLEVALCOORD1FVPROC) load(userptr, "glEvalCoord1fv");
    context->EvalCoord2d = (PFNGLEVALCOORD2DPROC) load(userptr, "glEvalCoord2d");
    context->EvalCoord2dv = (PFNGLEVALCOORD2DVPROC) load(userptr, "glEvalCoord2dv");
    context->EvalCoord2f = (PFNGLEVALCOORD2FPROC) load(userptr, "glEvalCoord2f");
    context->EvalCoord2fv = (PFNGLEVALCOORD2FVPROC) load(userptr, "glEvalCoord2fv");
    context->EvalMesh1 = (PFNGLEVALMESH1PROC) load(userptr, "glEvalMesh1");
    context->EvalMesh2 = (PFNGLEVALMESH2PROC) load(userptr, "glEvalMesh2");
    context->EvalPoint1 = (PFNGLEVALPOINT1PROC) load(userptr, "glEvalPoint1");
    context->EvalPoint2 = (PFNGLEVALPOINT2PROC) load(userptr, "glEvalPoint2");
    context->FeedbackBuffer = (PFNGLFEEDBACKBUFFERPROC) load(userptr, "glFeedbackBuffer");
    context->Finish = (PFNGLFINISHPROC) load(userptr, "glFinish");
    context->Flush = (PFNGLFLUSHPROC) load(userptr, "glFlush");
    context->Fogf = (PFNGLFOGFPROC) load(userptr, "glFogf");
    context->Fogfv = (PFNGLFOGFVPROC) load(userptr, "glFogfv");
    context->Fogi = (PFNGLFOGIPROC) load(userptr, "glFogi");
    context->Fogiv = (PFNGLFOGIVPROC) load(userptr, "glFogiv");
    context->FrontFace = (PFNGLFRONTFACEPROC) load(userptr, "glFrontFace");
    context->Frustum = (PFNGLFRUSTUMPROC) load(userptr, "glFrustum");
    context->GenLists = (PFNGLGENLISTSPROC) load(userptr, "glGenLists");
    context->GetBooleanv = (PFNGLGETBOOLEANVPROC) load(userptr, "glGetBooleanv");
    context->GetClipPlane = (PFNGLGETCLIPPLANEPROC) load(userptr, "glGetClipPlane");
    context->GetDoublev = (PFNGLGETDOUBLEVPROC) load(userptr, "glGetDoublev");
    context->GetError = (PFNGLGETERRORPROC) load(userptr, "glGetError");
    context->GetFloatv = (PFNGLGETFLOATVPROC) load(userptr, "glGetFloatv");
    context->GetIntegerv = (PFNGLGETINTEGERVPROC) load(userptr, "glGetIntegerv");
    context->GetLightfv = (PFNGLGETLIGHTFVPROC) load(userptr, "glGetLightfv");
    context->GetLightiv = (PFNGLGETLIGHTIVPROC) load(userptr, "glGetLightiv");
    context->GetMapdv = (PFNGLGETMAPDVPROC) load(userptr, "glGetMapdv");
    context->GetMapfv = (PFNGLGETMAPFVPROC) load(userptr, "glGetMapfv");
    context->GetMapiv = (PFNGLGETMAPIVPROC) load(userptr, "glGetMapiv");
    context->GetMaterialfv = (PFNGLGETMATERIALFVPROC) load(userptr, "glGetMaterialfv");
    context->GetMaterialiv = (PFNGLGETMATERIALIVPROC) load(userptr, "glGetMaterialiv");
    context->GetPixelMapfv = (PFNGLGETPIXELMAPFVPROC) load(userptr, "glGetPixelMapfv");
    context->GetPixelMapuiv = (PFNGLGETPIXELMAPUIVPROC) load(userptr, "glGetPixelMapuiv");
    context->GetPixelMapusv = (PFNGLGETPIXELMAPUSVPROC) load(userptr, "glGetPixelMapusv");
    context->GetPolygonStipple = (PFNGLGETPOLYGONSTIPPLEPROC) load(userptr, "glGetPolygonStipple");
    context->GetString = (PFNGLGETSTRINGPROC) load(userptr, "glGetString");
    context->GetTexEnvfv = (PFNGLGETTEXENVFVPROC) load(userptr, "glGetTexEnvfv");
    context->GetTexEnviv = (PFNGLGETTEXENVIVPROC) load(userptr, "glGetTexEnviv");
    context->GetTexGendv = (PFNGLGETTEXGENDVPROC) load(userptr, "glGetTexGendv");
    context->GetTexGenfv = (PFNGLGETTEXGENFVPROC) load(userptr, "glGetTexGenfv");
    context->GetTexGeniv = (PFNGLGETTEXGENIVPROC) load(userptr, "glGetTexGeniv");
    context->GetTexImage = (PFNGLGETTEXIMAGEPROC) load(userptr, "glGetTexImage");
    context->GetTexLevelParameterfv = (PFNGLGETTEXLEVELPARAMETERFVPROC) load(userptr, "glGetTexLevelParameterfv");
    context->GetTexLevelParameteriv = (PFNGLGETTEXLEVELPARAMETERIVPROC) load(userptr, "glGetTexLevelParameteriv");
    context->GetTexParameterfv = (PFNGLGETTEXPARAMETERFVPROC) load(userptr, "glGetTexParameterfv");
    context->GetTexParameteriv = (PFNGLGETTEXPARAMETERIVPROC) load(userptr, "glGetTexParameteriv");
    context->Hint = (PFNGLHINTPROC) load(userptr, "glHint");
    context->IndexMask = (PFNGLINDEXMASKPROC) load(userptr, "glIndexMask");
    context->Indexd = (PFNGLINDEXDPROC) load(userptr, "glIndexd");
    context->Indexdv = (PFNGLINDEXDVPROC) load(userptr, "glIndexdv");
    context->Indexf = (PFNGLINDEXFPROC) load(userptr, "glIndexf");
    context->Indexfv = (PFNGLINDEXFVPROC) load(userptr, "glIndexfv");
    context->Indexi = (PFNGLINDEXIPROC) load(userptr, "glIndexi");
    context->Indexiv = (PFNGLINDEXIVPROC) load(userptr, "glIndexiv");
    context->Indexs = (PFNGLINDEXSPROC) load(userptr, "glIndexs");
    context->Indexsv = (PFNGLINDEXSVPROC) load(userptr, "glIndexsv");
    context->InitNames = (PFNGLINITNAMESPROC) load(userptr, "glInitNames");
    context->IsEnabled = (PFNGLISENABLEDPROC) load(userptr, "glIsEnabled");
    context->IsList = (PFNGLISLISTPROC) load(userptr, "glIsList");
    context->LightModelf = (PFNGLLIGHTMODELFPROC) load(userptr, "glLightModelf");
    context->LightModelfv = (PFNGLLIGHTMODELFVPROC) load(userptr, "glLightModelfv");
    context->LightModeli = (PFNGLLIGHTMODELIPROC) load(userptr, "glLightModeli");
    context->LightModeliv = (PFNGLLIGHTMODELIVPROC) load(userptr, "glLightModeliv");
    context->Lightf = (PFNGLLIGHTFPROC) load(userptr, "glLightf");
    context->Lightfv = (PFNGLLIGHTFVPROC) load(userptr, "glLightfv");
    context->Lighti = (PFNGLLIGHTIPROC) load(userptr, "glLighti");
    context->Lightiv = (PFNGLLIGHTIVPROC) load(userptr, "glLightiv");
    context->LineStipple = (PFNGLLINESTIPPLEPROC) load(userptr, "glLineStipple");
    context->LineWidth = (PFNGLLINEWIDTHPROC) load(userptr, "glLineWidth");
    context->ListBase = (PFNGLLISTBASEPROC) load(userptr, "glListBase");
    context->LoadIdentity = (PFNGLLOADIDENTITYPROC) load(userptr, "glLoadIdentity");
    context->LoadMatrixd = (PFNGLLOADMATRIXDPROC) load(userptr, "glLoadMatrixd");
    context->LoadMatrixf = (PFNGLLOADMATRIXFPROC) load(userptr, "glLoadMatrixf");
    context->LoadName = (PFNGLLOADNAMEPROC) load(userptr, "glLoadName");
    context->LogicOp = (PFNGLLOGICOPPROC) load(userptr, "glLogicOp");
    context->Map1d = (PFNGLMAP1DPROC) load(userptr, "glMap1d");
    context->Map1f = (PFNGLMAP1FPROC) load(userptr, "glMap1f");
    context->Map2d = (PFNGLMAP2DPROC) load(userptr, "glMap2d");
    context->Map2f = (PFNGLMAP2FPROC) load(userptr, "glMap2f");
    context->MapGrid1d = (PFNGLMAPGRID1DPROC) load(userptr, "glMapGrid1d");
    context->MapGrid1f = (PFNGLMAPGRID1FPROC) load(userptr, "glMapGrid1f");
    context->MapGrid2d = (PFNGLMAPGRID2DPROC) load(userptr, "glMapGrid2d");
    context->MapGrid2f = (PFNGLMAPGRID2FPROC) load(userptr, "glMapGrid2f");
    context->Materialf = (PFNGLMATERIALFPROC) load(userptr, "glMaterialf");
    context->Materialfv = (PFNGLMATERIALFVPROC) load(userptr, "glMaterialfv");
    context->Materiali = (PFNGLMATERIALIPROC) load(userptr, "glMateriali");
    context->Materialiv = (PFNGLMATERIALIVPROC) load(userptr, "glMaterialiv");
    context->MatrixMode = (PFNGLMATRIXMODEPROC) load(userptr, "glMatrixMode");
    context->MultMatrixd = (PFNGLMULTMATRIXDPROC) load(userptr, "glMultMatrixd");
    context->MultMatrixf = (PFNGLMULTMATRIXFPROC) load(userptr, "glMultMatrixf");
    context->NewList = (PFNGLNEWLISTPROC) load(userptr, "glNewList");
    context->Normal3b = (PFNGLNORMAL3BPROC) load(userptr, "glNormal3b");
    context->Normal3bv = (PFNGLNORMAL3BVPROC) load(userptr, "glNormal3bv");
    context->Normal3d = (PFNGLNORMAL3DPROC) load(userptr, "glNormal3d");
    context->Normal3dv = (PFNGLNORMAL3DVPROC) load(userptr, "glNormal3dv");
    context->Normal3f = (PFNGLNORMAL3FPROC) load(userptr, "glNormal3f");
    context->Normal3fv = (PFNGLNORMAL3FVPROC) load(userptr, "glNormal3fv");
    context->Normal3i = (PFNGLNORMAL3IPROC) load(userptr, "glNormal3i");
    context->Normal3iv = (PFNGLNORMAL3IVPROC) load(userptr, "glNormal3iv");
    context->Normal3s = (PFNGLNORMAL3SPROC) load(userptr, "glNormal3s");
    context->Normal3sv = (PFNGLNORMAL3SVPROC) load(userptr, "glNormal3sv");
    context->Ortho = (PFNGLORTHOPROC) load(userptr, "glOrtho");
    context->PassThrough = (PFNGLPASSTHROUGHPROC) load(userptr, "glPassThrough");
    context->PixelMapfv = (PFNGLPIXELMAPFVPROC) load(userptr, "glPixelMapfv");
    context->PixelMapuiv = (PFNGLPIXELMAPUIVPROC) load(userptr, "glPixelMapuiv");
    context->PixelMapusv = (PFNGLPIXELMAPUSVPROC) load(userptr, "glPixelMapusv");
    context->PixelStoref = (PFNGLPIXELSTOREFPROC) load(userptr, "glPixelStoref");
    context->PixelStorei = (PFNGLPIXELSTOREIPROC) load(userptr, "glPixelStorei");
    context->PixelTransferf = (PFNGLPIXELTRANSFERFPROC) load(userptr, "glPixelTransferf");
    context->PixelTransferi = (PFNGLPIXELTRANSFERIPROC) load(userptr, "glPixelTransferi");
    context->PixelZoom = (PFNGLPIXELZOOMPROC) load(userptr, "glPixelZoom");
    context->PointSize = (PFNGLPOINTSIZEPROC) load(userptr, "glPointSize");
    context->PolygonMode = (PFNGLPOLYGONMODEPROC) load(userptr, "glPolygonMode");
    context->PolygonStipple = (PFNGLPOLYGONSTIPPLEPROC) load(userptr, "glPolygonStipple");
    context->PopAttrib = (PFNGLPOPATTRIBPROC) load(userptr, "glPopAttrib");
    context->PopMatrix = (PFNGLPOPMATRIXPROC) load(userptr, "glPopMatrix");
    context->PopName = (PFNGLPOPNAMEPROC) load(userptr, "glPopName");
    context->PushAttrib = (PFNGLPUSHATTRIBPROC) load(userptr, "glPushAttrib");
    context->PushMatrix = (PFNGLPUSHMATRIXPROC) load(userptr, "glPushMatrix");
    context->PushName = (PFNGLPUSHNAMEPROC) load(userptr, "glPushName");
    context->RasterPos2d = (PFNGLRASTERPOS2DPROC) load(userptr, "glRasterPos2d");
    context->RasterPos2dv = (PFNGLRASTERPOS2DVPROC) load(userptr, "glRasterPos2dv");
    context->RasterPos2f = (PFNGLRASTERPOS2FPROC) load(userptr, "glRasterPos2f");
    context->RasterPos2fv = (PFNGLRASTERPOS2FVPROC) load(userptr, "glRasterPos2fv");
    context->RasterPos2i = (PFNGLRASTERPOS2IPROC) load(userptr, "glRasterPos2i");
    context->RasterPos2iv = (PFNGLRASTERPOS2IVPROC) load(userptr, "glRasterPos2iv");
    context->RasterPos2s = (PFNGLRASTERPOS2SPROC) load(userptr, "glRasterPos2s");
    context->RasterPos2sv = (PFNGLRASTERPOS2SVPROC) load(userptr, "glRasterPos2sv");
    context->RasterPos3d = (PFNGLRASTERPOS3DPROC) load(userptr, "glRasterPos3d");
    context->RasterPos3dv = (PFNGLRASTERPOS3DVPROC) load(userptr, "glRasterPos3dv");
    context->RasterPos3f = (PFNGLRASTERPOS3FPROC) load(userptr, "glRasterPos3f");
    context->RasterPos3fv = (PFNGLRASTERPOS3FVPROC) load(userptr, "glRasterPos3fv");
    context->RasterPos3i = (PFNGLRASTERPOS3IPROC) load(userptr, "glRasterPos3i");
    context->RasterPos3iv = (PFNGLRASTERPOS3IVPROC) load(userptr, "glRasterPos3iv");
    context->RasterPos3s = (PFNGLRASTERPOS3SPROC) load(userptr, "glRasterPos3s");
    context->RasterPos3sv = (PFNGLRASTERPOS3SVPROC) load(userptr, "glRasterPos3sv");
    context->RasterPos4d = (PFNGLRASTERPOS4DPROC) load(userptr, "glRasterPos4d");
    context->RasterPos4dv = (PFNGLRASTERPOS4DVPROC) load(userptr, "glRasterPos4dv");
    context->RasterPos4f = (PFNGLRASTERPOS4FPROC) load(userptr, "glRasterPos4f");
    context->RasterPos4fv = (PFNGLRASTERPOS4FVPROC) load(userptr, "glRasterPos4fv");
    context->RasterPos4i = (PFNGLRASTERPOS4IPROC) load(userptr, "glRasterPos4i");
    context->RasterPos4iv = (PFNGLRASTERPOS4IVPROC) load(userptr, "glRasterPos4iv");
    context->RasterPos4s = (PFNGLRASTERPOS4SPROC) load(userptr, "glRasterPos4s");
    context->RasterPos4sv = (PFNGLRASTERPOS4SVPROC) load(userptr, "glRasterPos4sv");
    context->ReadBuffer = (PFNGLREADBUFFERPROC) load(userptr, "glReadBuffer");
    context->ReadPixels = (PFNGLREADPIXELSPROC) load(userptr, "glReadPixels");
    context->Rectd = (PFNGLRECTDPROC) load(userptr, "glRectd");
    context->Rectdv = (PFNGLRECTDVPROC) load(userptr, "glRectdv");
    context->Rectf = (PFNGLRECTFPROC) load(userptr, "glRectf");
    context->Rectfv = (PFNGLRECTFVPROC) load(userptr, "glRectfv");
    context->Recti = (PFNGLRECTIPROC) load(userptr, "glRecti");
    context->Rectiv = (PFNGLRECTIVPROC) load(userptr, "glRectiv");
    context->Rects = (PFNGLRECTSPROC) load(userptr, "glRects");
    context->Rectsv = (PFNGLRECTSVPROC) load(userptr, "glRectsv");
    context->RenderMode = (PFNGLRENDERMODEPROC) load(userptr, "glRenderMode");
    context->Rotated = (PFNGLROTATEDPROC) load(userptr, "glRotated");
    context->Rotatef = (PFNGLROTATEFPROC) load(userptr, "glRotatef");
    context->Scaled = (PFNGLSCALEDPROC) load(userptr, "glScaled");
    context->Scalef = (PFNGLSCALEFPROC) load(userptr, "glScalef");
    context->Scissor = (PFNGLSCISSORPROC) load(userptr, "glScissor");
    context->SelectBuffer = (PFNGLSELECTBUFFERPROC) load(userptr, "glSelectBuffer");
    context->ShadeModel = (PFNGLSHADEMODELPROC) load(userptr, "glShadeModel");
    context->StencilFunc = (PFNGLSTENCILFUNCPROC) load(userptr, "glStencilFunc");
    context->StencilMask = (PFNGLSTENCILMASKPROC) load(userptr, "glStencilMask");
    context->StencilOp = (PFNGLSTENCILOPPROC) load(userptr, "glStencilOp");
    context->TexCoord1d = (PFNGLTEXCOORD1DPROC) load(userptr, "glTexCoord1d");
    context->TexCoord1dv = (PFNGLTEXCOORD1DVPROC) load(userptr, "glTexCoord1dv");
    context->TexCoord1f = (PFNGLTEXCOORD1FPROC) load(userptr, "glTexCoord1f");
    context->TexCoord1fv = (PFNGLTEXCOORD1FVPROC) load(userptr, "glTexCoord1fv");
    context->TexCoord1i = (PFNGLTEXCOORD1IPROC) load(userptr, "glTexCoord1i");
    context->TexCoord1iv = (PFNGLTEXCOORD1IVPROC) load(userptr, "glTexCoord1iv");
    context->TexCoord1s = (PFNGLTEXCOORD1SPROC) load(userptr, "glTexCoord1s");
    context->TexCoord1sv = (PFNGLTEXCOORD1SVPROC) load(userptr, "glTexCoord1sv");
    context->TexCoord2d = (PFNGLTEXCOORD2DPROC) load(userptr, "glTexCoord2d");
    context->TexCoord2dv = (PFNGLTEXCOORD2DVPROC) load(userptr, "glTexCoord2dv");
    context->TexCoord2f = (PFNGLTEXCOORD2FPROC) load(userptr, "glTexCoord2f");
    context->TexCoord2fv = (PFNGLTEXCOORD2FVPROC) load(userptr, "glTexCoord2fv");
    context->TexCoord2i = (PFNGLTEXCOORD2IPROC) load(userptr, "glTexCoord2i");
    context->TexCoord2iv = (PFNGLTEXCOORD2IVPROC) load(userptr, "glTexCoord2iv");
    context->TexCoord2s = (PFNGLTEXCOORD2SPROC) load(userptr, "glTexCoord2s");
    context->TexCoord2sv = (PFNGLTEXCOORD2SVPROC) load(userptr, "glTexCoord2sv");
    context->TexCoord3d = (PFNGLTEXCOORD3DPROC) load(userptr, "glTexCoord3d");
    context->TexCoord3dv = (PFNGLTEXCOORD3DVPROC) load(userptr, "glTexCoord3dv");
    context->TexCoord3f = (PFNGLTEXCOORD3FPROC) load(userptr, "glTexCoord3f");
    context->TexCoord3fv = (PFNGLTEXCOORD3FVPROC) load(userptr, "glTexCoord3fv");
    context->TexCoord3i = (PFNGLTEXCOORD3IPROC) load(userptr, "glTexCoord3i");
    context->TexCoord3iv = (PFNGLTEXCOORD3IVPROC) load(userptr, "glTexCoord3iv");
    context->TexCoord3s = (PFNGLTEXCOORD3SPROC) load(userptr, "glTexCoord3s");
    context->TexCoord3sv = (PFNGLTEXCOORD3SVPROC) load(userptr, "glTexCoord3sv");
    context->TexCoord4d = (PFNGLTEXCOORD4DPROC) load(userptr, "glTexCoord4d");
    context->TexCoord4dv = (PFNGLTEXCOORD4DVPROC) load(userptr, "glTexCoord4dv");
    context->TexCoord4f = (PFNGLTEXCOORD4FPROC) load(userptr, "glTexCoord4f");
    context->TexCoord4fv = (PFNGLTEXCOORD4FVPROC) load(userptr, "glTexCoord4fv");
    context->TexCoord4i = (PFNGLTEXCOORD4IPROC) load(userptr, "glTexCoord4i");
    context->TexCoord4iv = (PFNGLTEXCOORD4IVPROC) load(userptr, "glTexCoord4iv");
    context->TexCoord4s = (PFNGLTEXCOORD4SPROC) load(userptr, "glTexCoord4s");
    context->TexCoord4sv = (PFNGLTEXCOORD4SVPROC) load(userptr, "glTexCoord4sv");
    context->TexEnvf = (PFNGLTEXENVFPROC) load(userptr, "glTexEnvf");
    context->TexEnvfv = (PFNGLTEXENVFVPROC) load(userptr, "glTexEnvfv");
    context->TexEnvi = (PFNGLTEXENVIPROC) load(userptr, "glTexEnvi");
    context->TexEnviv = (PFNGLTEXENVIVPROC) load(userptr, "glTexEnviv");
    context->TexGend = (PFNGLTEXGENDPROC) load(userptr, "glTexGend");
    context->TexGendv = (PFNGLTEXGENDVPROC) load(userptr, "glTexGendv");
    context->TexGenf = (PFNGLTEXGENFPROC) load(userptr, "glTexGenf");
    context->TexGenfv = (PFNGLTEXGENFVPROC) load(userptr, "glTexGenfv");
    context->TexGeni = (PFNGLTEXGENIPROC) load(userptr, "glTexGeni");
    context->TexGeniv = (PFNGLTEXGENIVPROC) load(userptr, "glTexGeniv");
    context->TexImage1D = (PFNGLTEXIMAGE1DPROC) load(userptr, "glTexImage1D");
    context->TexImage2D = (PFNGLTEXIMAGE2DPROC) load(userptr, "glTexImage2D");
    context->TexParameterf = (PFNGLTEXPARAMETERFPROC) load(userptr, "glTexParameterf");
    context->TexParameterfv = (PFNGLTEXPARAMETERFVPROC) load(userptr, "glTexParameterfv");
    context->TexParameteri = (PFNGLTEXPARAMETERIPROC) load(userptr, "glTexParameteri");
    context->TexParameteriv = (PFNGLTEXPARAMETERIVPROC) load(userptr, "glTexParameteriv");
    context->Translated = (PFNGLTRANSLATEDPROC) load(userptr, "glTranslated");
    context->Translatef = (PFNGLTRANSLATEFPROC) load(userptr, "glTranslatef");
    context->Vertex2d = (PFNGLVERTEX2DPROC) load(userptr, "glVertex2d");
    context->Vertex2dv = (PFNGLVERTEX2DVPROC) load(userptr, "glVertex2dv");
    context->Vertex2f = (PFNGLVERTEX2FPROC) load(userptr, "glVertex2f");
    context->Vertex2fv = (PFNGLVERTEX2FVPROC) load(userptr, "glVertex2fv");
    context->Vertex2i = (PFNGLVERTEX2IPROC) load(userptr, "glVertex2i");
    context->Vertex2iv = (PFNGLVERTEX2IVPROC) load(userptr, "glVertex2iv");
    context->Vertex2s = (PFNGLVERTEX2SPROC) load(userptr, "glVertex2s");
    context->Vertex2sv = (PFNGLVERTEX2SVPROC) load(userptr, "glVertex2sv");
    context->Vertex3d = (PFNGLVERTEX3DPROC) load(userptr, "glVertex3d");
    context->Vertex3dv = (PFNGLVERTEX3DVPROC) load(userptr, "glVertex3dv");
    context->Vertex3f = (PFNGLVERTEX3FPROC) load(userptr, "glVertex3f");
    context->Vertex3fv = (PFNGLVERTEX3FVPROC) load(userptr, "glVertex3fv");
    context->Vertex3i = (PFNGLVERTEX3IPROC) load(userptr, "glVertex3i");
    context->Vertex3iv = (PFNGLVERTEX3IVPROC) load(userptr, "glVertex3iv");
    context->Vertex3s = (PFNGLVERTEX3SPROC) load(userptr, "glVertex3s");
    context->Vertex3sv = (PFNGLVERTEX3SVPROC) load(userptr, "glVertex3sv");
    context->Vertex4d = (PFNGLVERTEX4DPROC) load(userptr, "glVertex4d");
    context->Vertex4dv = (PFNGLVERTEX4DVPROC) load(userptr, "glVertex4dv");
    context->Vertex4f = (PFNGLVERTEX4FPROC) load(userptr, "glVertex4f");
    context->Vertex4fv = (PFNGLVERTEX4FVPROC) load(userptr, "glVertex4fv");
    context->Vertex4i = (PFNGLVERTEX4IPROC) load(userptr, "glVertex4i");
    context->Vertex4iv = (PFNGLVERTEX4IVPROC) load(userptr, "glVertex4iv");
    context->Vertex4s = (PFNGLVERTEX4SPROC) load(userptr, "glVertex4s");
    context->Vertex4sv = (PFNGLVERTEX4SVPROC) load(userptr, "glVertex4sv");
    context->Viewport = (PFNGLVIEWPORTPROC) load(userptr, "glViewport");
}
static void glad_gl_load_GL_VERSION_1_1(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_1_1) return;
    context->AreTexturesResident = (PFNGLARETEXTURESRESIDENTPROC) load(userptr, "glAreTexturesResident");
    context->ArrayElement = (PFNGLARRAYELEMENTPROC) load(userptr, "glArrayElement");
    context->BindTexture = (PFNGLBINDTEXTUREPROC) load(userptr, "glBindTexture");
    context->ColorPointer = (PFNGLCOLORPOINTERPROC) load(userptr, "glColorPointer");
    context->CopyTexImage1D = (PFNGLCOPYTEXIMAGE1DPROC) load(userptr, "glCopyTexImage1D");
    context->CopyTexImage2D = (PFNGLCOPYTEXIMAGE2DPROC) load(userptr, "glCopyTexImage2D");
    context->CopyTexSubImage1D = (PFNGLCOPYTEXSUBIMAGE1DPROC) load(userptr, "glCopyTexSubImage1D");
    context->CopyTexSubImage2D = (PFNGLCOPYTEXSUBIMAGE2DPROC) load(userptr, "glCopyTexSubImage2D");
    context->DeleteTextures = (PFNGLDELETETEXTURESPROC) load(userptr, "glDeleteTextures");
    context->DisableClientState = (PFNGLDISABLECLIENTSTATEPROC) load(userptr, "glDisableClientState");
    context->DrawArrays = (PFNGLDRAWARRAYSPROC) load(userptr, "glDrawArrays");
    context->DrawElements = (PFNGLDRAWELEMENTSPROC) load(userptr, "glDrawElements");
    context->EdgeFlagPointer = (PFNGLEDGEFLAGPOINTERPROC) load(userptr, "glEdgeFlagPointer");
    context->EnableClientState = (PFNGLENABLECLIENTSTATEPROC) load(userptr, "glEnableClientState");
    context->GenTextures = (PFNGLGENTEXTURESPROC) load(userptr, "glGenTextures");
    context->GetPointerv = (PFNGLGETPOINTERVPROC) load(userptr, "glGetPointerv");
    context->IndexPointer = (PFNGLINDEXPOINTERPROC) load(userptr, "glIndexPointer");
    context->Indexub = (PFNGLINDEXUBPROC) load(userptr, "glIndexub");
    context->Indexubv = (PFNGLINDEXUBVPROC) load(userptr, "glIndexubv");
    context->InterleavedArrays = (PFNGLINTERLEAVEDARRAYSPROC) load(userptr, "glInterleavedArrays");
    context->IsTexture = (PFNGLISTEXTUREPROC) load(userptr, "glIsTexture");
    context->NormalPointer = (PFNGLNORMALPOINTERPROC) load(userptr, "glNormalPointer");
    context->PolygonOffset = (PFNGLPOLYGONOFFSETPROC) load(userptr, "glPolygonOffset");
    context->PopClientAttrib = (PFNGLPOPCLIENTATTRIBPROC) load(userptr, "glPopClientAttrib");
    context->PrioritizeTextures = (PFNGLPRIORITIZETEXTURESPROC) load(userptr, "glPrioritizeTextures");
    context->PushClientAttrib = (PFNGLPUSHCLIENTATTRIBPROC) load(userptr, "glPushClientAttrib");
    context->TexCoordPointer = (PFNGLTEXCOORDPOINTERPROC) load(userptr, "glTexCoordPointer");
    context->TexSubImage1D = (PFNGLTEXSUBIMAGE1DPROC) load(userptr, "glTexSubImage1D");
    context->TexSubImage2D = (PFNGLTEXSUBIMAGE2DPROC) load(userptr, "glTexSubImage2D");
    context->VertexPointer = (PFNGLVERTEXPOINTERPROC) load(userptr, "glVertexPointer");
}
static void glad_gl_load_GL_VERSION_1_2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_1_2) return;
    context->CopyTexSubImage3D = (PFNGLCOPYTEXSUBIMAGE3DPROC) load(userptr, "glCopyTexSubImage3D");
    context->DrawRangeElements = (PFNGLDRAWRANGEELEMENTSPROC) load(userptr, "glDrawRangeElements");
    context->TexImage3D = (PFNGLTEXIMAGE3DPROC) load(userptr, "glTexImage3D");
    context->TexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC) load(userptr, "glTexSubImage3D");
}
static void glad_gl_load_GL_VERSION_1_3(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_1_3) return;
    context->ActiveTexture = (PFNGLACTIVETEXTUREPROC) load(userptr, "glActiveTexture");
    context->ClientActiveTexture = (PFNGLCLIENTACTIVETEXTUREPROC) load(userptr, "glClientActiveTexture");
    context->CompressedTexImage1D = (PFNGLCOMPRESSEDTEXIMAGE1DPROC) load(userptr, "glCompressedTexImage1D");
    context->CompressedTexImage2D = (PFNGLCOMPRESSEDTEXIMAGE2DPROC) load(userptr, "glCompressedTexImage2D");
    context->CompressedTexImage3D = (PFNGLCOMPRESSEDTEXIMAGE3DPROC) load(userptr, "glCompressedTexImage3D");
    context->CompressedTexSubImage1D = (PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC) load(userptr, "glCompressedTexSubImage1D");
    context->CompressedTexSubImage2D = (PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC) load(userptr, "glCompressedTexSubImage2D");
    context->CompressedTexSubImage3D = (PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC) load(userptr, "glCompressedTexSubImage3D");
    context->GetCompressedTexImage = (PFNGLGETCOMPRESSEDTEXIMAGEPROC) load(userptr, "glGetCompressedTexImage");
    context->LoadTransposeMatrixd = (PFNGLLOADTRANSPOSEMATRIXDPROC) load(userptr, "glLoadTransposeMatrixd");
    context->LoadTransposeMatrixf = (PFNGLLOADTRANSPOSEMATRIXFPROC) load(userptr, "glLoadTransposeMatrixf");
    context->MultTransposeMatrixd = (PFNGLMULTTRANSPOSEMATRIXDPROC) load(userptr, "glMultTransposeMatrixd");
    context->MultTransposeMatrixf = (PFNGLMULTTRANSPOSEMATRIXFPROC) load(userptr, "glMultTransposeMatrixf");
    context->MultiTexCoord1d = (PFNGLMULTITEXCOORD1DPROC) load(userptr, "glMultiTexCoord1d");
    context->MultiTexCoord1dv = (PFNGLMULTITEXCOORD1DVPROC) load(userptr, "glMultiTexCoord1dv");
    context->MultiTexCoord1f = (PFNGLMULTITEXCOORD1FPROC) load(userptr, "glMultiTexCoord1f");
    context->MultiTexCoord1fv = (PFNGLMULTITEXCOORD1FVPROC) load(userptr, "glMultiTexCoord1fv");
    context->MultiTexCoord1i = (PFNGLMULTITEXCOORD1IPROC) load(userptr, "glMultiTexCoord1i");
    context->MultiTexCoord1iv = (PFNGLMULTITEXCOORD1IVPROC) load(userptr, "glMultiTexCoord1iv");
    context->MultiTexCoord1s = (PFNGLMULTITEXCOORD1SPROC) load(userptr, "glMultiTexCoord1s");
    context->MultiTexCoord1sv = (PFNGLMULTITEXCOORD1SVPROC) load(userptr, "glMultiTexCoord1sv");
    context->MultiTexCoord2d = (PFNGLMULTITEXCOORD2DPROC) load(userptr, "glMultiTexCoord2d");
    context->MultiTexCoord2dv = (PFNGLMULTITEXCOORD2DVPROC) load(userptr, "glMultiTexCoord2dv");
    context->MultiTexCoord2f = (PFNGLMULTITEXCOORD2FPROC) load(userptr, "glMultiTexCoord2f");
    context->MultiTexCoord2fv = (PFNGLMULTITEXCOORD2FVPROC) load(userptr, "glMultiTexCoord2fv");
    context->MultiTexCoord2i = (PFNGLMULTITEXCOORD2IPROC) load(userptr, "glMultiTexCoord2i");
    context->MultiTexCoord2iv = (PFNGLMULTITEXCOORD2IVPROC) load(userptr, "glMultiTexCoord2iv");
    context->MultiTexCoord2s = (PFNGLMULTITEXCOORD2SPROC) load(userptr, "glMultiTexCoord2s");
    context->MultiTexCoord2sv = (PFNGLMULTITEXCOORD2SVPROC) load(userptr, "glMultiTexCoord2sv");
    context->MultiTexCoord3d = (PFNGLMULTITEXCOORD3DPROC) load(userptr, "glMultiTexCoord3d");
    context->MultiTexCoord3dv = (PFNGLMULTITEXCOORD3DVPROC) load(userptr, "glMultiTexCoord3dv");
    context->MultiTexCoord3f = (PFNGLMULTITEXCOORD3FPROC) load(userptr, "glMultiTexCoord3f");
    context->MultiTexCoord3fv = (PFNGLMULTITEXCOORD3FVPROC) load(userptr, "glMultiTexCoord3fv");
    context->MultiTexCoord3i = (PFNGLMULTITEXCOORD3IPROC) load(userptr, "glMultiTexCoord3i");
    context->MultiTexCoord3iv = (PFNGLMULTITEXCOORD3IVPROC) load(userptr, "glMultiTexCoord3iv");
    context->MultiTexCoord3s = (PFNGLMULTITEXCOORD3SPROC) load(userptr, "glMultiTexCoord3s");
    context->MultiTexCoord3sv = (PFNGLMULTITEXCOORD3SVPROC) load(userptr, "glMultiTexCoord3sv");
    context->MultiTexCoord4d = (PFNGLMULTITEXCOORD4DPROC) load(userptr, "glMultiTexCoord4d");
    context->MultiTexCoord4dv = (PFNGLMULTITEXCOORD4DVPROC) load(userptr, "glMultiTexCoord4dv");
    context->MultiTexCoord4f = (PFNGLMULTITEXCOORD4FPROC) load(userptr, "glMultiTexCoord4f");
    context->MultiTexCoord4fv = (PFNGLMULTITEXCOORD4FVPROC) load(userptr, "glMultiTexCoord4fv");
    context->MultiTexCoord4i = (PFNGLMULTITEXCOORD4IPROC) load(userptr, "glMultiTexCoord4i");
    context->MultiTexCoord4iv = (PFNGLMULTITEXCOORD4IVPROC) load(userptr, "glMultiTexCoord4iv");
    context->MultiTexCoord4s = (PFNGLMULTITEXCOORD4SPROC) load(userptr, "glMultiTexCoord4s");
    context->MultiTexCoord4sv = (PFNGLMULTITEXCOORD4SVPROC) load(userptr, "glMultiTexCoord4sv");
    context->SampleCoverage = (PFNGLSAMPLECOVERAGEPROC) load(userptr, "glSampleCoverage");
}
static void glad_gl_load_GL_VERSION_1_4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_1_4) return;
    context->BlendColor = (PFNGLBLENDCOLORPROC) load(userptr, "glBlendColor");
    context->BlendEquation = (PFNGLBLENDEQUATIONPROC) load(userptr, "glBlendEquation");
    context->BlendFuncSeparate = (PFNGLBLENDFUNCSEPARATEPROC) load(userptr, "glBlendFuncSeparate");
    context->FogCoordPointer = (PFNGLFOGCOORDPOINTERPROC) load(userptr, "glFogCoordPointer");
    context->FogCoordd = (PFNGLFOGCOORDDPROC) load(userptr, "glFogCoordd");
    context->FogCoorddv = (PFNGLFOGCOORDDVPROC) load(userptr, "glFogCoorddv");
    context->FogCoordf = (PFNGLFOGCOORDFPROC) load(userptr, "glFogCoordf");
    context->FogCoordfv = (PFNGLFOGCOORDFVPROC) load(userptr, "glFogCoordfv");
    context->MultiDrawArrays = (PFNGLMULTIDRAWARRAYSPROC) load(userptr, "glMultiDrawArrays");
    context->MultiDrawElements = (PFNGLMULTIDRAWELEMENTSPROC) load(userptr, "glMultiDrawElements");
    context->PointParameterf = (PFNGLPOINTPARAMETERFPROC) load(userptr, "glPointParameterf");
    context->PointParameterfv = (PFNGLPOINTPARAMETERFVPROC) load(userptr, "glPointParameterfv");
    context->PointParameteri = (PFNGLPOINTPARAMETERIPROC) load(userptr, "glPointParameteri");
    context->PointParameteriv = (PFNGLPOINTPARAMETERIVPROC) load(userptr, "glPointParameteriv");
    context->SecondaryColor3b = (PFNGLSECONDARYCOLOR3BPROC) load(userptr, "glSecondaryColor3b");
    context->SecondaryColor3bv = (PFNGLSECONDARYCOLOR3BVPROC) load(userptr, "glSecondaryColor3bv");
    context->SecondaryColor3d = (PFNGLSECONDARYCOLOR3DPROC) load(userptr, "glSecondaryColor3d");
    context->SecondaryColor3dv = (PFNGLSECONDARYCOLOR3DVPROC) load(userptr, "glSecondaryColor3dv");
    context->SecondaryColor3f = (PFNGLSECONDARYCOLOR3FPROC) load(userptr, "glSecondaryColor3f");
    context->SecondaryColor3fv = (PFNGLSECONDARYCOLOR3FVPROC) load(userptr, "glSecondaryColor3fv");
    context->SecondaryColor3i = (PFNGLSECONDARYCOLOR3IPROC) load(userptr, "glSecondaryColor3i");
    context->SecondaryColor3iv = (PFNGLSECONDARYCOLOR3IVPROC) load(userptr, "glSecondaryColor3iv");
    context->SecondaryColor3s = (PFNGLSECONDARYCOLOR3SPROC) load(userptr, "glSecondaryColor3s");
    context->SecondaryColor3sv = (PFNGLSECONDARYCOLOR3SVPROC) load(userptr, "glSecondaryColor3sv");
    context->SecondaryColor3ub = (PFNGLSECONDARYCOLOR3UBPROC) load(userptr, "glSecondaryColor3ub");
    context->SecondaryColor3ubv = (PFNGLSECONDARYCOLOR3UBVPROC) load(userptr, "glSecondaryColor3ubv");
    context->SecondaryColor3ui = (PFNGLSECONDARYCOLOR3UIPROC) load(userptr, "glSecondaryColor3ui");
    context->SecondaryColor3uiv = (PFNGLSECONDARYCOLOR3UIVPROC) load(userptr, "glSecondaryColor3uiv");
    context->SecondaryColor3us = (PFNGLSECONDARYCOLOR3USPROC) load(userptr, "glSecondaryColor3us");
    context->SecondaryColor3usv = (PFNGLSECONDARYCOLOR3USVPROC) load(userptr, "glSecondaryColor3usv");
    context->SecondaryColorPointer = (PFNGLSECONDARYCOLORPOINTERPROC) load(userptr, "glSecondaryColorPointer");
    context->WindowPos2d = (PFNGLWINDOWPOS2DPROC) load(userptr, "glWindowPos2d");
    context->WindowPos2dv = (PFNGLWINDOWPOS2DVPROC) load(userptr, "glWindowPos2dv");
    context->WindowPos2f = (PFNGLWINDOWPOS2FPROC) load(userptr, "glWindowPos2f");
    context->WindowPos2fv = (PFNGLWINDOWPOS2FVPROC) load(userptr, "glWindowPos2fv");
    context->WindowPos2i = (PFNGLWINDOWPOS2IPROC) load(userptr, "glWindowPos2i");
    context->WindowPos2iv = (PFNGLWINDOWPOS2IVPROC) load(userptr, "glWindowPos2iv");
    context->WindowPos2s = (PFNGLWINDOWPOS2SPROC) load(userptr, "glWindowPos2s");
    context->WindowPos2sv = (PFNGLWINDOWPOS2SVPROC) load(userptr, "glWindowPos2sv");
    context->WindowPos3d = (PFNGLWINDOWPOS3DPROC) load(userptr, "glWindowPos3d");
    context->WindowPos3dv = (PFNGLWINDOWPOS3DVPROC) load(userptr, "glWindowPos3dv");
    context->WindowPos3f = (PFNGLWINDOWPOS3FPROC) load(userptr, "glWindowPos3f");
    context->WindowPos3fv = (PFNGLWINDOWPOS3FVPROC) load(userptr, "glWindowPos3fv");
    context->WindowPos3i = (PFNGLWINDOWPOS3IPROC) load(userptr, "glWindowPos3i");
    context->WindowPos3iv = (PFNGLWINDOWPOS3IVPROC) load(userptr, "glWindowPos3iv");
    context->WindowPos3s = (PFNGLWINDOWPOS3SPROC) load(userptr, "glWindowPos3s");
    context->WindowPos3sv = (PFNGLWINDOWPOS3SVPROC) load(userptr, "glWindowPos3sv");
}
static void glad_gl_load_GL_VERSION_1_5(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_1_5) return;
    context->BeginQuery = (PFNGLBEGINQUERYPROC) load(userptr, "glBeginQuery");
    context->BindBuffer = (PFNGLBINDBUFFERPROC) load(userptr, "glBindBuffer");
    context->BufferData = (PFNGLBUFFERDATAPROC) load(userptr, "glBufferData");
    context->BufferSubData = (PFNGLBUFFERSUBDATAPROC) load(userptr, "glBufferSubData");
    context->DeleteBuffers = (PFNGLDELETEBUFFERSPROC) load(userptr, "glDeleteBuffers");
    context->DeleteQueries = (PFNGLDELETEQUERIESPROC) load(userptr, "glDeleteQueries");
    context->EndQuery = (PFNGLENDQUERYPROC) load(userptr, "glEndQuery");
    context->GenBuffers = (PFNGLGENBUFFERSPROC) load(userptr, "glGenBuffers");
    context->GenQueries = (PFNGLGENQUERIESPROC) load(userptr, "glGenQueries");
    context->GetBufferParameteriv = (PFNGLGETBUFFERPARAMETERIVPROC) load(userptr, "glGetBufferParameteriv");
    context->GetBufferPointerv = (PFNGLGETBUFFERPOINTERVPROC) load(userptr, "glGetBufferPointerv");
    context->GetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC) load(userptr, "glGetBufferSubData");
    context->GetQueryObjectiv = (PFNGLGETQUERYOBJECTIVPROC) load(userptr, "glGetQueryObjectiv");
    context->GetQueryObjectuiv = (PFNGLGETQUERYOBJECTUIVPROC) load(userptr, "glGetQueryObjectuiv");
    context->GetQueryiv = (PFNGLGETQUERYIVPROC) load(userptr, "glGetQueryiv");
    context->IsBuffer = (PFNGLISBUFFERPROC) load(userptr, "glIsBuffer");
    context->IsQuery = (PFNGLISQUERYPROC) load(userptr, "glIsQuery");
    context->MapBuffer = (PFNGLMAPBUFFERPROC) load(userptr, "glMapBuffer");
    context->UnmapBuffer = (PFNGLUNMAPBUFFERPROC) load(userptr, "glUnmapBuffer");
}
static void glad_gl_load_GL_VERSION_2_0(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_2_0) return;
    context->AttachShader = (PFNGLATTACHSHADERPROC) load(userptr, "glAttachShader");
    context->BindAttribLocation = (PFNGLBINDATTRIBLOCATIONPROC) load(userptr, "glBindAttribLocation");
    context->BlendEquationSeparate = (PFNGLBLENDEQUATIONSEPARATEPROC) load(userptr, "glBlendEquationSeparate");
    context->CompileShader = (PFNGLCOMPILESHADERPROC) load(userptr, "glCompileShader");
    context->CreateProgram = (PFNGLCREATEPROGRAMPROC) load(userptr, "glCreateProgram");
    context->CreateShader = (PFNGLCREATESHADERPROC) load(userptr, "glCreateShader");
    context->DeleteProgram = (PFNGLDELETEPROGRAMPROC) load(userptr, "glDeleteProgram");
    context->DeleteShader = (PFNGLDELETESHADERPROC) load(userptr, "glDeleteShader");
    context->DetachShader = (PFNGLDETACHSHADERPROC) load(userptr, "glDetachShader");
    context->DisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC) load(userptr, "glDisableVertexAttribArray");
    context->DrawBuffers = (PFNGLDRAWBUFFERSPROC) load(userptr, "glDrawBuffers");
    context->EnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC) load(userptr, "glEnableVertexAttribArray");
    context->GetActiveAttrib = (PFNGLGETACTIVEATTRIBPROC) load(userptr, "glGetActiveAttrib");
    context->GetActiveUniform = (PFNGLGETACTIVEUNIFORMPROC) load(userptr, "glGetActiveUniform");
    context->GetAttachedShaders = (PFNGLGETATTACHEDSHADERSPROC) load(userptr, "glGetAttachedShaders");
    context->GetAttribLocation = (PFNGLGETATTRIBLOCATIONPROC) load(userptr, "glGetAttribLocation");
    context->GetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC) load(userptr, "glGetProgramInfoLog");
    context->GetProgramiv = (PFNGLGETPROGRAMIVPROC) load(userptr, "glGetProgramiv");
    context->GetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC) load(userptr, "glGetShaderInfoLog");
    context->GetShaderSource = (PFNGLGETSHADERSOURCEPROC) load(userptr, "glGetShaderSource");
    context->GetShaderiv = (PFNGLGETSHADERIVPROC) load(userptr, "glGetShaderiv");
    context->GetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC) load(userptr, "glGetUniformLocation");
    context->GetUniformfv = (PFNGLGETUNIFORMFVPROC) load(userptr, "glGetUniformfv");
    context->GetUniformiv = (PFNGLGETUNIFORMIVPROC) load(userptr, "glGetUniformiv");
    context->GetVertexAttribPointerv = (PFNGLGETVERTEXATTRIBPOINTERVPROC) load(userptr, "glGetVertexAttribPointerv");
    context->GetVertexAttribdv = (PFNGLGETVERTEXATTRIBDVPROC) load(userptr, "glGetVertexAttribdv");
    context->GetVertexAttribfv = (PFNGLGETVERTEXATTRIBFVPROC) load(userptr, "glGetVertexAttribfv");
    context->GetVertexAttribiv = (PFNGLGETVERTEXATTRIBIVPROC) load(userptr, "glGetVertexAttribiv");
    context->IsProgram = (PFNGLISPROGRAMPROC) load(userptr, "glIsProgram");
    context->IsShader = (PFNGLISSHADERPROC) load(userptr, "glIsShader");
    context->LinkProgram = (PFNGLLINKPROGRAMPROC) load(userptr, "glLinkProgram");
    context->ShaderSource = (PFNGLSHADERSOURCEPROC) load(userptr, "glShaderSource");
    context->StencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATEPROC) load(userptr, "glStencilFuncSeparate");
    context->StencilMaskSeparate = (PFNGLSTENCILMASKSEPARATEPROC) load(userptr, "glStencilMaskSeparate");
    context->StencilOpSeparate = (PFNGLSTENCILOPSEPARATEPROC) load(userptr, "glStencilOpSeparate");
    context->Uniform1f = (PFNGLUNIFORM1FPROC) load(userptr, "glUniform1f");
    context->Uniform1fv = (PFNGLUNIFORM1FVPROC) load(userptr, "glUniform1fv");
    context->Uniform1i = (PFNGLUNIFORM1IPROC) load(userptr, "glUniform1i");
    context->Uniform1iv = (PFNGLUNIFORM1IVPROC) load(userptr, "glUniform1iv");
    context->Uniform2f = (PFNGLUNIFORM2FPROC) load(userptr, "glUniform2f");
    context->Uniform2fv = (PFNGLUNIFORM2FVPROC) load(userptr, "glUniform2fv");
    context->Uniform2i = (PFNGLUNIFORM2IPROC) load(userptr, "glUniform2i");
    context->Uniform2iv = (PFNGLUNIFORM2IVPROC) load(userptr, "glUniform2iv");
    context->Uniform3f = (PFNGLUNIFORM3FPROC) load(userptr, "glUniform3f");
    context->Uniform3fv = (PFNGLUNIFORM3FVPROC) load(userptr, "glUniform3fv");
    context->Uniform3i = (PFNGLUNIFORM3IPROC) load(userptr, "glUniform3i");
    context->Uniform3iv = (PFNGLUNIFORM3IVPROC) load(userptr, "glUniform3iv");
    context->Uniform4f = (PFNGLUNIFORM4FPROC) load(userptr, "glUniform4f");
    context->Uniform4fv = (PFNGLUNIFORM4FVPROC) load(userptr, "glUniform4fv");
    context->Uniform4i = (PFNGLUNIFORM4IPROC) load(userptr, "glUniform4i");
    context->Uniform4iv = (PFNGLUNIFORM4IVPROC) load(userptr, "glUniform4iv");
    context->UniformMatrix2fv = (PFNGLUNIFORMMATRIX2FVPROC) load(userptr, "glUniformMatrix2fv");
    context->UniformMatrix3fv = (PFNGLUNIFORMMATRIX3FVPROC) load(userptr, "glUniformMatrix3fv");
    context->UniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC) load(userptr, "glUniformMatrix4fv");
    context->UseProgram = (PFNGLUSEPROGRAMPROC) load(userptr, "glUseProgram");
    context->ValidateProgram = (PFNGLVALIDATEPROGRAMPROC) load(userptr, "glValidateProgram");
    context->VertexAttrib1d = (PFNGLVERTEXATTRIB1DPROC) load(userptr, "glVertexAttrib1d");
    context->VertexAttrib1dv = (PFNGLVERTEXATTRIB1DVPROC) load(userptr, "glVertexAttrib1dv");
    context->VertexAttrib1f = (PFNGLVERTEXATTRIB1FPROC) load(userptr, "glVertexAttrib1f");
    context->VertexAttrib1fv = (PFNGLVERTEXATTRIB1FVPROC) load(userptr, "glVertexAttrib1fv");
    context->VertexAttrib1s = (PFNGLVERTEXATTRIB1SPROC) load(userptr, "glVertexAttrib1s");
    context->VertexAttrib1sv = (PFNGLVERTEXATTRIB1SVPROC) load(userptr, "glVertexAttrib1sv");
    context->VertexAttrib2d = (PFNGLVERTEXATTRIB2DPROC) load(userptr, "glVertexAttrib2d");
    context->VertexAttrib2dv = (PFNGLVERTEXATTRIB2DVPROC) load(userptr, "glVertexAttrib2dv");
    context->VertexAttrib2f = (PFNGLVERTEXATTRIB2FPROC) load(userptr, "glVertexAttrib2f");
    context->VertexAttrib2fv = (PFNGLVERTEXATTRIB2FVPROC) load(userptr, "glVertexAttrib2fv");
    context->VertexAttrib2s = (PFNGLVERTEXATTRIB2SPROC) load(userptr, "glVertexAttrib2s");
    context->VertexAttrib2sv = (PFNGLVERTEXATTRIB2SVPROC) load(userptr, "glVertexAttrib2sv");
    context->VertexAttrib3d = (PFNGLVERTEXATTRIB3DPROC) load(userptr, "glVertexAttrib3d");
    context->VertexAttrib3dv = (PFNGLVERTEXATTRIB3DVPROC) load(userptr, "glVertexAttrib3dv");
    context->VertexAttrib3f = (PFNGLVERTEXATTRIB3FPROC) load(userptr, "glVertexAttrib3f");
    context->VertexAttrib3fv = (PFNGLVERTEXATTRIB3FVPROC) load(userptr, "glVertexAttrib3fv");
    context->VertexAttrib3s = (PFNGLVERTEXATTRIB3SPROC) load(userptr, "glVertexAttrib3s");
    context->VertexAttrib3sv = (PFNGLVERTEXATTRIB3SVPROC) load(userptr, "glVertexAttrib3sv");
    context->VertexAttrib4Nbv = (PFNGLVERTEXATTRIB4NBVPROC) load(userptr, "glVertexAttrib4Nbv");
    context->VertexAttrib4Niv = (PFNGLVERTEXATTRIB4NIVPROC) load(userptr, "glVertexAttrib4Niv");
    context->VertexAttrib4Nsv = (PFNGLVERTEXATTRIB4NSVPROC) load(userptr, "glVertexAttrib4Nsv");
    context->VertexAttrib4Nub = (PFNGLVERTEXATTRIB4NUBPROC) load(userptr, "glVertexAttrib4Nub");
    context->VertexAttrib4Nubv = (PFNGLVERTEXATTRIB4NUBVPROC) load(userptr, "glVertexAttrib4Nubv");
    context->VertexAttrib4Nuiv = (PFNGLVERTEXATTRIB4NUIVPROC) load(userptr, "glVertexAttrib4Nuiv");
    context->VertexAttrib4Nusv = (PFNGLVERTEXATTRIB4NUSVPROC) load(userptr, "glVertexAttrib4Nusv");
    context->VertexAttrib4bv = (PFNGLVERTEXATTRIB4BVPROC) load(userptr, "glVertexAttrib4bv");
    context->VertexAttrib4d = (PFNGLVERTEXATTRIB4DPROC) load(userptr, "glVertexAttrib4d");
    context->VertexAttrib4dv = (PFNGLVERTEXATTRIB4DVPROC) load(userptr, "glVertexAttrib4dv");
    context->VertexAttrib4f = (PFNGLVERTEXATTRIB4FPROC) load(userptr, "glVertexAttrib4f");
    context->VertexAttrib4fv = (PFNGLVERTEXATTRIB4FVPROC) load(userptr, "glVertexAttrib4fv");
    context->VertexAttrib4iv = (PFNGLVERTEXATTRIB4IVPROC) load(userptr, "glVertexAttrib4iv");
    context->VertexAttrib4s = (PFNGLVERTEXATTRIB4SPROC) load(userptr, "glVertexAttrib4s");
    context->VertexAttrib4sv = (PFNGLVERTEXATTRIB4SVPROC) load(userptr, "glVertexAttrib4sv");
    context->VertexAttrib4ubv = (PFNGLVERTEXATTRIB4UBVPROC) load(userptr, "glVertexAttrib4ubv");
    context->VertexAttrib4uiv = (PFNGLVERTEXATTRIB4UIVPROC) load(userptr, "glVertexAttrib4uiv");
    context->VertexAttrib4usv = (PFNGLVERTEXATTRIB4USVPROC) load(userptr, "glVertexAttrib4usv");
    context->VertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC) load(userptr, "glVertexAttribPointer");
}
static void glad_gl_load_GL_VERSION_2_1(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_2_1) return;
    context->UniformMatrix2x3fv = (PFNGLUNIFORMMATRIX2X3FVPROC) load(userptr, "glUniformMatrix2x3fv");
    context->UniformMatrix2x4fv = (PFNGLUNIFORMMATRIX2X4FVPROC) load(userptr, "glUniformMatrix2x4fv");
    context->UniformMatrix3x2fv = (PFNGLUNIFORMMATRIX3X2FVPROC) load(userptr, "glUniformMatrix3x2fv");
    context->UniformMatrix3x4fv = (PFNGLUNIFORMMATRIX3X4FVPROC) load(userptr, "glUniformMatrix3x4fv");
    context->UniformMatrix4x2fv = (PFNGLUNIFORMMATRIX4X2FVPROC) load(userptr, "glUniformMatrix4x2fv");
    context->UniformMatrix4x3fv = (PFNGLUNIFORMMATRIX4X3FVPROC) load(userptr, "glUniformMatrix4x3fv");
}
static void glad_gl_load_GL_VERSION_3_0(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_3_0) return;
    context->BeginConditionalRender = (PFNGLBEGINCONDITIONALRENDERPROC) load(userptr, "glBeginConditionalRender");
    context->BeginTransformFeedback = (PFNGLBEGINTRANSFORMFEEDBACKPROC) load(userptr, "glBeginTransformFeedback");
    context->BindBufferBase = (PFNGLBINDBUFFERBASEPROC) load(userptr, "glBindBufferBase");
    context->BindBufferRange = (PFNGLBINDBUFFERRANGEPROC) load(userptr, "glBindBufferRange");
    context->BindFragDataLocation = (PFNGLBINDFRAGDATALOCATIONPROC) load(userptr, "glBindFragDataLocation");
    context->BindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC) load(userptr, "glBindFramebuffer");
    context->BindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC) load(userptr, "glBindRenderbuffer");
    context->BindVertexArray = (PFNGLBINDVERTEXARRAYPROC) load(userptr, "glBindVertexArray");
    context->BlitFramebuffer = (PFNGLBLITFRAMEBUFFERPROC) load(userptr, "glBlitFramebuffer");
    context->CheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC) load(userptr, "glCheckFramebufferStatus");
    context->ClampColor = (PFNGLCLAMPCOLORPROC) load(userptr, "glClampColor");
    context->ClearBufferfi = (PFNGLCLEARBUFFERFIPROC) load(userptr, "glClearBufferfi");
    context->ClearBufferfv = (PFNGLCLEARBUFFERFVPROC) load(userptr, "glClearBufferfv");
    context->ClearBufferiv = (PFNGLCLEARBUFFERIVPROC) load(userptr, "glClearBufferiv");
    context->ClearBufferuiv = (PFNGLCLEARBUFFERUIVPROC) load(userptr, "glClearBufferuiv");
    context->ColorMaski = (PFNGLCOLORMASKIPROC) load(userptr, "glColorMaski");
    context->DeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC) load(userptr, "glDeleteFramebuffers");
    context->DeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC) load(userptr, "glDeleteRenderbuffers");
    context->DeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC) load(userptr, "glDeleteVertexArrays");
    context->Disablei = (PFNGLDISABLEIPROC) load(userptr, "glDisablei");
    context->Enablei = (PFNGLENABLEIPROC) load(userptr, "glEnablei");
    context->EndConditionalRender = (PFNGLENDCONDITIONALRENDERPROC) load(userptr, "glEndConditionalRender");
    context->EndTransformFeedback = (PFNGLENDTRANSFORMFEEDBACKPROC) load(userptr, "glEndTransformFeedback");
    context->FlushMappedBufferRange = (PFNGLFLUSHMAPPEDBUFFERRANGEPROC) load(userptr, "glFlushMappedBufferRange");
    context->FramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC) load(userptr, "glFramebufferRenderbuffer");
    context->FramebufferTexture1D = (PFNGLFRAMEBUFFERTEXTURE1DPROC) load(userptr, "glFramebufferTexture1D");
    context->FramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC) load(userptr, "glFramebufferTexture2D");
    context->FramebufferTexture3D = (PFNGLFRAMEBUFFERTEXTURE3DPROC) load(userptr, "glFramebufferTexture3D");
    context->FramebufferTextureLayer = (PFNGLFRAMEBUFFERTEXTURELAYERPROC) load(userptr, "glFramebufferTextureLayer");
    context->GenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC) load(userptr, "glGenFramebuffers");
    context->GenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC) load(userptr, "glGenRenderbuffers");
    context->GenVertexArrays = (PFNGLGENVERTEXARRAYSPROC) load(userptr, "glGenVertexArrays");
    context->GenerateMipmap = (PFNGLGENERATEMIPMAPPROC) load(userptr, "glGenerateMipmap");
    context->GetBooleani_v = (PFNGLGETBOOLEANI_VPROC) load(userptr, "glGetBooleani_v");
    context->GetFragDataLocation = (PFNGLGETFRAGDATALOCATIONPROC) load(userptr, "glGetFragDataLocation");
    context->GetFramebufferAttachmentParameteriv = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC) load(userptr, "glGetFramebufferAttachmentParameteriv");
    context->GetIntegeri_v = (PFNGLGETINTEGERI_VPROC) load(userptr, "glGetIntegeri_v");
    context->GetRenderbufferParameteriv = (PFNGLGETRENDERBUFFERPARAMETERIVPROC) load(userptr, "glGetRenderbufferParameteriv");
    context->GetStringi = (PFNGLGETSTRINGIPROC) load(userptr, "glGetStringi");
    context->GetTexParameterIiv = (PFNGLGETTEXPARAMETERIIVPROC) load(userptr, "glGetTexParameterIiv");
    context->GetTexParameterIuiv = (PFNGLGETTEXPARAMETERIUIVPROC) load(userptr, "glGetTexParameterIuiv");
    context->GetTransformFeedbackVarying = (PFNGLGETTRANSFORMFEEDBACKVARYINGPROC) load(userptr, "glGetTransformFeedbackVarying");
    context->GetUniformuiv = (PFNGLGETUNIFORMUIVPROC) load(userptr, "glGetUniformuiv");
    context->GetVertexAttribIiv = (PFNGLGETVERTEXATTRIBIIVPROC) load(userptr, "glGetVertexAttribIiv");
    context->GetVertexAttribIuiv = (PFNGLGETVERTEXATTRIBIUIVPROC) load(userptr, "glGetVertexAttribIuiv");
    context->IsEnabledi = (PFNGLISENABLEDIPROC) load(userptr, "glIsEnabledi");
    context->IsFramebuffer = (PFNGLISFRAMEBUFFERPROC) load(userptr, "glIsFramebuffer");
    context->IsRenderbuffer = (PFNGLISRENDERBUFFERPROC) load(userptr, "glIsRenderbuffer");
    context->IsVertexArray = (PFNGLISVERTEXARRAYPROC) load(userptr, "glIsVertexArray");
    context->MapBufferRange = (PFNGLMAPBUFFERRANGEPROC) load(userptr, "glMapBufferRange");
    context->RenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC) load(userptr, "glRenderbufferStorage");
    context->RenderbufferStorageMultisample = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC) load(userptr, "glRenderbufferStorageMultisample");
    context->TexParameterIiv = (PFNGLTEXPARAMETERIIVPROC) load(userptr, "glTexParameterIiv");
    context->TexParameterIuiv = (PFNGLTEXPARAMETERIUIVPROC) load(userptr, "glTexParameterIuiv");
    context->TransformFeedbackVaryings = (PFNGLTRANSFORMFEEDBACKVARYINGSPROC) load(userptr, "glTransformFeedbackVaryings");
    context->Uniform1ui = (PFNGLUNIFORM1UIPROC) load(userptr, "glUniform1ui");
    context->Uniform1uiv = (PFNGLUNIFORM1UIVPROC) load(userptr, "glUniform1uiv");
    context->Uniform2ui = (PFNGLUNIFORM2UIPROC) load(userptr, "glUniform2ui");
    context->Uniform2uiv = (PFNGLUNIFORM2UIVPROC) load(userptr, "glUniform2uiv");
    context->Uniform3ui = (PFNGLUNIFORM3UIPROC) load(userptr, "glUniform3ui");
    context->Uniform3uiv = (PFNGLUNIFORM3UIVPROC) load(userptr, "glUniform3uiv");
    context->Uniform4ui = (PFNGLUNIFORM4UIPROC) load(userptr, "glUniform4ui");
    context->Uniform4uiv = (PFNGLUNIFORM4UIVPROC) load(userptr, "glUniform4uiv");
    context->VertexAttribI1i = (PFNGLVERTEXATTRIBI1IPROC) load(userptr, "glVertexAttribI1i");
    context->VertexAttribI1iv = (PFNGLVERTEXATTRIBI1IVPROC) load(userptr, "glVertexAttribI1iv");
    context->VertexAttribI1ui = (PFNGLVERTEXATTRIBI1UIPROC) load(userptr, "glVertexAttribI1ui");
    context->VertexAttribI1uiv = (PFNGLVERTEXATTRIBI1UIVPROC) load(userptr, "glVertexAttribI1uiv");
    context->VertexAttribI2i = (PFNGLVERTEXATTRIBI2IPROC) load(userptr, "glVertexAttribI2i");
    context->VertexAttribI2iv = (PFNGLVERTEXATTRIBI2IVPROC) load(userptr, "glVertexAttribI2iv");
    context->VertexAttribI2ui = (PFNGLVERTEXATTRIBI2UIPROC) load(userptr, "glVertexAttribI2ui");
    context->VertexAttribI2uiv = (PFNGLVERTEXATTRIBI2UIVPROC) load(userptr, "glVertexAttribI2uiv");
    context->VertexAttribI3i = (PFNGLVERTEXATTRIBI3IPROC) load(userptr, "glVertexAttribI3i");
    context->VertexAttribI3iv = (PFNGLVERTEXATTRIBI3IVPROC) load(userptr, "glVertexAttribI3iv");
    context->VertexAttribI3ui = (PFNGLVERTEXATTRIBI3UIPROC) load(userptr, "glVertexAttribI3ui");
    context->VertexAttribI3uiv = (PFNGLVERTEXATTRIBI3UIVPROC) load(userptr, "glVertexAttribI3uiv");
    context->VertexAttribI4bv = (PFNGLVERTEXATTRIBI4BVPROC) load(userptr, "glVertexAttribI4bv");
    context->VertexAttribI4i = (PFNGLVERTEXATTRIBI4IPROC) load(userptr, "glVertexAttribI4i");
    context->VertexAttribI4iv = (PFNGLVERTEXATTRIBI4IVPROC) load(userptr, "glVertexAttribI4iv");
    context->VertexAttribI4sv = (PFNGLVERTEXATTRIBI4SVPROC) load(userptr, "glVertexAttribI4sv");
    context->VertexAttribI4ubv = (PFNGLVERTEXATTRIBI4UBVPROC) load(userptr, "glVertexAttribI4ubv");
    context->VertexAttribI4ui = (PFNGLVERTEXATTRIBI4UIPROC) load(userptr, "glVertexAttribI4ui");
    context->VertexAttribI4uiv = (PFNGLVERTEXATTRIBI4UIVPROC) load(userptr, "glVertexAttribI4uiv");
    context->VertexAttribI4usv = (PFNGLVERTEXATTRIBI4USVPROC) load(userptr, "glVertexAttribI4usv");
    context->VertexAttribIPointer = (PFNGLVERTEXATTRIBIPOINTERPROC) load(userptr, "glVertexAttribIPointer");
}
static void glad_gl_load_GL_VERSION_3_1(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_3_1) return;
    context->BindBufferBase = (PFNGLBINDBUFFERBASEPROC) load(userptr, "glBindBufferBase");
    context->BindBufferRange = (PFNGLBINDBUFFERRANGEPROC) load(userptr, "glBindBufferRange");
    context->CopyBufferSubData = (PFNGLCOPYBUFFERSUBDATAPROC) load(userptr, "glCopyBufferSubData");
    context->DrawArraysInstanced = (PFNGLDRAWARRAYSINSTANCEDPROC) load(userptr, "glDrawArraysInstanced");
    context->DrawElementsInstanced = (PFNGLDRAWELEMENTSINSTANCEDPROC) load(userptr, "glDrawElementsInstanced");
    context->GetActiveUniformBlockName = (PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC) load(userptr, "glGetActiveUniformBlockName");
    context->GetActiveUniformBlockiv = (PFNGLGETACTIVEUNIFORMBLOCKIVPROC) load(userptr, "glGetActiveUniformBlockiv");
    context->GetActiveUniformName = (PFNGLGETACTIVEUNIFORMNAMEPROC) load(userptr, "glGetActiveUniformName");
    context->GetActiveUniformsiv = (PFNGLGETACTIVEUNIFORMSIVPROC) load(userptr, "glGetActiveUniformsiv");
    context->GetIntegeri_v = (PFNGLGETINTEGERI_VPROC) load(userptr, "glGetIntegeri_v");
    context->GetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC) load(userptr, "glGetUniformBlockIndex");
    context->GetUniformIndices = (PFNGLGETUNIFORMINDICESPROC) load(userptr, "glGetUniformIndices");
    context->PrimitiveRestartIndex = (PFNGLPRIMITIVERESTARTINDEXPROC) load(userptr, "glPrimitiveRestartIndex");
    context->TexBuffer = (PFNGLTEXBUFFERPROC) load(userptr, "glTexBuffer");
    context->UniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC) load(userptr, "glUniformBlockBinding");
}
static void glad_gl_load_GL_VERSION_3_2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_3_2) return;
    context->ClientWaitSync = (PFNGLCLIENTWAITSYNCPROC) load(userptr, "glClientWaitSync");
    context->DeleteSync = (PFNGLDELETESYNCPROC) load(userptr, "glDeleteSync");
    context->DrawElementsBaseVertex = (PFNGLDRAWELEMENTSBASEVERTEXPROC) load(userptr, "glDrawElementsBaseVertex");
    context->DrawElementsInstancedBaseVertex = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC) load(userptr, "glDrawElementsInstancedBaseVertex");
    context->DrawRangeElementsBaseVertex = (PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC) load(userptr, "glDrawRangeElementsBaseVertex");
    context->FenceSync = (PFNGLFENCESYNCPROC) load(userptr, "glFenceSync");
    context->FramebufferTexture = (PFNGLFRAMEBUFFERTEXTUREPROC) load(userptr, "glFramebufferTexture");
    context->GetBufferParameteri64v = (PFNGLGETBUFFERPARAMETERI64VPROC) load(userptr, "glGetBufferParameteri64v");
    context->GetInteger64i_v = (PFNGLGETINTEGER64I_VPROC) load(userptr, "glGetInteger64i_v");
    context->GetInteger64v = (PFNGLGETINTEGER64VPROC) load(userptr, "glGetInteger64v");
    context->GetMultisamplefv = (PFNGLGETMULTISAMPLEFVPROC) load(userptr, "glGetMultisamplefv");
    context->GetSynciv = (PFNGLGETSYNCIVPROC) load(userptr, "glGetSynciv");
    context->IsSync = (PFNGLISSYNCPROC) load(userptr, "glIsSync");
    context->MultiDrawElementsBaseVertex = (PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC) load(userptr, "glMultiDrawElementsBaseVertex");
    context->ProvokingVertex = (PFNGLPROVOKINGVERTEXPROC) load(userptr, "glProvokingVertex");
    context->SampleMaski = (PFNGLSAMPLEMASKIPROC) load(userptr, "glSampleMaski");
    context->TexImage2DMultisample = (PFNGLTEXIMAGE2DMULTISAMPLEPROC) load(userptr, "glTexImage2DMultisample");
    context->TexImage3DMultisample = (PFNGLTEXIMAGE3DMULTISAMPLEPROC) load(userptr, "glTexImage3DMultisample");
    context->WaitSync = (PFNGLWAITSYNCPROC) load(userptr, "glWaitSync");
}
static void glad_gl_load_GL_VERSION_3_3(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_3_3) return;
    context->BindFragDataLocationIndexed = (PFNGLBINDFRAGDATALOCATIONINDEXEDPROC) load(userptr, "glBindFragDataLocationIndexed");
    context->BindSampler = (PFNGLBINDSAMPLERPROC) load(userptr, "glBindSampler");
    context->ColorP3ui = (PFNGLCOLORP3UIPROC) load(userptr, "glColorP3ui");
    context->ColorP3uiv = (PFNGLCOLORP3UIVPROC) load(userptr, "glColorP3uiv");
    context->ColorP4ui = (PFNGLCOLORP4UIPROC) load(userptr, "glColorP4ui");
    context->ColorP4uiv = (PFNGLCOLORP4UIVPROC) load(userptr, "glColorP4uiv");
    context->DeleteSamplers = (PFNGLDELETESAMPLERSPROC) load(userptr, "glDeleteSamplers");
    context->GenSamplers = (PFNGLGENSAMPLERSPROC) load(userptr, "glGenSamplers");
    context->GetFragDataIndex = (PFNGLGETFRAGDATAINDEXPROC) load(userptr, "glGetFragDataIndex");
    context->GetQueryObjecti64v = (PFNGLGETQUERYOBJECTI64VPROC) load(userptr, "glGetQueryObjecti64v");
    context->GetQueryObjectui64v = (PFNGLGETQUERYOBJECTUI64VPROC) load(userptr, "glGetQueryObjectui64v");
    context->GetSamplerParameterIiv = (PFNGLGETSAMPLERPARAMETERIIVPROC) load(userptr, "glGetSamplerParameterIiv");
    context->GetSamplerParameterIuiv = (PFNGLGETSAMPLERPARAMETERIUIVPROC) load(userptr, "glGetSamplerParameterIuiv");
    context->GetSamplerParameterfv = (PFNGLGETSAMPLERPARAMETERFVPROC) load(userptr, "glGetSamplerParameterfv");
    context->GetSamplerParameteriv = (PFNGLGETSAMPLERPARAMETERIVPROC) load(userptr, "glGetSamplerParameteriv");
    context->IsSampler = (PFNGLISSAMPLERPROC) load(userptr, "glIsSampler");
    context->MultiTexCoordP1ui = (PFNGLMULTITEXCOORDP1UIPROC) load(userptr, "glMultiTexCoordP1ui");
    context->MultiTexCoordP1uiv = (PFNGLMULTITEXCOORDP1UIVPROC) load(userptr, "glMultiTexCoordP1uiv");
    context->MultiTexCoordP2ui = (PFNGLMULTITEXCOORDP2UIPROC) load(userptr, "glMultiTexCoordP2ui");
    context->MultiTexCoordP2uiv = (PFNGLMULTITEXCOORDP2UIVPROC) load(userptr, "glMultiTexCoordP2uiv");
    context->MultiTexCoordP3ui = (PFNGLMULTITEXCOORDP3UIPROC) load(userptr, "glMultiTexCoordP3ui");
    context->MultiTexCoordP3uiv = (PFNGLMULTITEXCOORDP3UIVPROC) load(userptr, "glMultiTexCoordP3uiv");
    context->MultiTexCoordP4ui = (PFNGLMULTITEXCOORDP4UIPROC) load(userptr, "glMultiTexCoordP4ui");
    context->MultiTexCoordP4uiv = (PFNGLMULTITEXCOORDP4UIVPROC) load(userptr, "glMultiTexCoordP4uiv");
    context->NormalP3ui = (PFNGLNORMALP3UIPROC) load(userptr, "glNormalP3ui");
    context->NormalP3uiv = (PFNGLNORMALP3UIVPROC) load(userptr, "glNormalP3uiv");
    context->QueryCounter = (PFNGLQUERYCOUNTERPROC) load(userptr, "glQueryCounter");
    context->SamplerParameterIiv = (PFNGLSAMPLERPARAMETERIIVPROC) load(userptr, "glSamplerParameterIiv");
    context->SamplerParameterIuiv = (PFNGLSAMPLERPARAMETERIUIVPROC) load(userptr, "glSamplerParameterIuiv");
    context->SamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC) load(userptr, "glSamplerParameterf");
    context->SamplerParameterfv = (PFNGLSAMPLERPARAMETERFVPROC) load(userptr, "glSamplerParameterfv");
    context->SamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC) load(userptr, "glSamplerParameteri");
    context->SamplerParameteriv = (PFNGLSAMPLERPARAMETERIVPROC) load(userptr, "glSamplerParameteriv");
    context->SecondaryColorP3ui = (PFNGLSECONDARYCOLORP3UIPROC) load(userptr, "glSecondaryColorP3ui");
    context->SecondaryColorP3uiv = (PFNGLSECONDARYCOLORP3UIVPROC) load(userptr, "glSecondaryColorP3uiv");
    context->TexCoordP1ui = (PFNGLTEXCOORDP1UIPROC) load(userptr, "glTexCoordP1ui");
    context->TexCoordP1uiv = (PFNGLTEXCOORDP1UIVPROC) load(userptr, "glTexCoordP1uiv");
    context->TexCoordP2ui = (PFNGLTEXCOORDP2UIPROC) load(userptr, "glTexCoordP2ui");
    context->TexCoordP2uiv = (PFNGLTEXCOORDP2UIVPROC) load(userptr, "glTexCoordP2uiv");
    context->TexCoordP3ui = (PFNGLTEXCOORDP3UIPROC) load(userptr, "glTexCoordP3ui");
    context->TexCoordP3uiv = (PFNGLTEXCOORDP3UIVPROC) load(userptr, "glTexCoordP3uiv");
    context->TexCoordP4ui = (PFNGLTEXCOORDP4UIPROC) load(userptr, "glTexCoordP4ui");
    context->TexCoordP4uiv = (PFNGLTEXCOORDP4UIVPROC) load(userptr, "glTexCoordP4uiv");
    context->VertexAttribDivisor = (PFNGLVERTEXATTRIBDIVISORPROC) load(userptr, "glVertexAttribDivisor");
    context->VertexAttribP1ui = (PFNGLVERTEXATTRIBP1UIPROC) load(userptr, "glVertexAttribP1ui");
    context->VertexAttribP1uiv = (PFNGLVERTEXATTRIBP1UIVPROC) load(userptr, "glVertexAttribP1uiv");
    context->VertexAttribP2ui = (PFNGLVERTEXATTRIBP2UIPROC) load(userptr, "glVertexAttribP2ui");
    context->VertexAttribP2uiv = (PFNGLVERTEXATTRIBP2UIVPROC) load(userptr, "glVertexAttribP2uiv");
    context->VertexAttribP3ui = (PFNGLVERTEXATTRIBP3UIPROC) load(userptr, "glVertexAttribP3ui");
    context->VertexAttribP3uiv = (PFNGLVERTEXATTRIBP3UIVPROC) load(userptr, "glVertexAttribP3uiv");
    context->VertexAttribP4ui = (PFNGLVERTEXATTRIBP4UIPROC) load(userptr, "glVertexAttribP4ui");
    context->VertexAttribP4uiv = (PFNGLVERTEXATTRIBP4UIVPROC) load(userptr, "glVertexAttribP4uiv");
    context->VertexP2ui = (PFNGLVERTEXP2UIPROC) load(userptr, "glVertexP2ui");
    context->VertexP2uiv = (PFNGLVERTEXP2UIVPROC) load(userptr, "glVertexP2uiv");
    context->VertexP3ui = (PFNGLVERTEXP3UIPROC) load(userptr, "glVertexP3ui");
    context->VertexP3uiv = (PFNGLVERTEXP3UIVPROC) load(userptr, "glVertexP3uiv");
    context->VertexP4ui = (PFNGLVERTEXP4UIPROC) load(userptr, "glVertexP4ui");
    context->VertexP4uiv = (PFNGLVERTEXP4UIVPROC) load(userptr, "glVertexP4uiv");
}
static void glad_gl_load_GL_VERSION_4_0(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_4_0) return;
    context->BeginQueryIndexed = (PFNGLBEGINQUERYINDEXEDPROC) load(userptr, "glBeginQueryIndexed");
    context->BindTransformFeedback = (PFNGLBINDTRANSFORMFEEDBACKPROC) load(userptr, "glBindTransformFeedback");
    context->BlendEquationSeparatei = (PFNGLBLENDEQUATIONSEPARATEIPROC) load(userptr, "glBlendEquationSeparatei");
    context->BlendEquationi = (PFNGLBLENDEQUATIONIPROC) load(userptr, "glBlendEquationi");
    context->BlendFuncSeparatei = (PFNGLBLENDFUNCSEPARATEIPROC) load(userptr, "glBlendFuncSeparatei");
    context->BlendFunci = (PFNGLBLENDFUNCIPROC) load(userptr, "glBlendFunci");
    context->DeleteTransformFeedbacks = (PFNGLDELETETRANSFORMFEEDBACKSPROC) load(userptr, "glDeleteTransformFeedbacks");
    context->DrawArraysIndirect = (PFNGLDRAWARRAYSINDIRECTPROC) load(userptr, "glDrawArraysIndirect");
    context->DrawElementsIndirect = (PFNGLDRAWELEMENTSINDIRECTPROC) load(userptr, "glDrawElementsIndirect");
    context->DrawTransformFeedback = (PFNGLDRAWTRANSFORMFEEDBACKPROC) load(userptr, "glDrawTransformFeedback");
    context->DrawTransformFeedbackStream = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC) load(userptr, "glDrawTransformFeedbackStream");
    context->EndQueryIndexed = (PFNGLENDQUERYINDEXEDPROC) load(userptr, "glEndQueryIndexed");
    context->GenTransformFeedbacks = (PFNGLGENTRANSFORMFEEDBACKSPROC) load(userptr, "glGenTransformFeedbacks");
    context->GetActiveSubroutineName = (PFNGLGETACTIVESUBROUTINENAMEPROC) load(userptr, "glGetActiveSubroutineName");
    context->GetActiveSubroutineUniformName = (PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC) load(userptr, "glGetActiveSubroutineUniformName");
    context->GetActiveSubroutineUniformiv = (PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC) load(userptr, "glGetActiveSubroutineUniformiv");
    context->GetProgramStageiv = (PFNGLGETPROGRAMSTAGEIVPROC) load(userptr, "glGetProgramStageiv");
    context->GetQueryIndexediv = (PFNGLGETQUERYINDEXEDIVPROC) load(userptr, "glGetQueryIndexediv");
    context->GetSubroutineIndex = (PFNGLGETSUBROUTINEINDEXPROC) load(userptr, "glGetSubroutineIndex");
    context->GetSubroutineUniformLocation = (PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC) load(userptr, "glGetSubroutineUniformLocation");
    context->GetUniformSubroutineuiv = (PFNGLGETUNIFORMSUBROUTINEUIVPROC) load(userptr, "glGetUniformSubroutineuiv");
    context->GetUniformdv = (PFNGLGETUNIFORMDVPROC) load(userptr, "glGetUniformdv");
    context->IsTransformFeedback = (PFNGLISTRANSFORMFEEDBACKPROC) load(userptr, "glIsTransformFeedback");
    context->MinSampleShading = (PFNGLMINSAMPLESHADINGPROC) load(userptr, "glMinSampleShading");
    context->PatchParameterfv = (PFNGLPATCHPARAMETERFVPROC) load(userptr, "glPatchParameterfv");
    context->PatchParameteri = (PFNGLPATCHPARAMETERIPROC) load(userptr, "glPatchParameteri");
    context->PauseTransformFeedback = (PFNGLPAUSETRANSFORMFEEDBACKPROC) load(userptr, "glPauseTransformFeedback");
    context->ResumeTransformFeedback = (PFNGLRESUMETRANSFORMFEEDBACKPROC) load(userptr, "glResumeTransformFeedback");
    context->Uniform1d = (PFNGLUNIFORM1DPROC) load(userptr, "glUniform1d");
    context->Uniform1dv = (PFNGLUNIFORM1DVPROC) load(userptr, "glUniform1dv");
    context->Uniform2d = (PFNGLUNIFORM2DPROC) load(userptr, "glUniform2d");
    context->Uniform2dv = (PFNGLUNIFORM2DVPROC) load(userptr, "glUniform2dv");
    context->Uniform3d = (PFNGLUNIFORM3DPROC) load(userptr, "glUniform3d");
    context->Uniform3dv = (PFNGLUNIFORM3DVPROC) load(userptr, "glUniform3dv");
    context->Uniform4d = (PFNGLUNIFORM4DPROC) load(userptr, "glUniform4d");
    context->Uniform4dv = (PFNGLUNIFORM4DVPROC) load(userptr, "glUniform4dv");
    context->UniformMatrix2dv = (PFNGLUNIFORMMATRIX2DVPROC) load(userptr, "glUniformMatrix2dv");
    context->UniformMatrix2x3dv = (PFNGLUNIFORMMATRIX2X3DVPROC) load(userptr, "glUniformMatrix2x3dv");
    context->UniformMatrix2x4dv = (PFNGLUNIFORMMATRIX2X4DVPROC) load(userptr, "glUniformMatrix2x4dv");
    context->UniformMatrix3dv = (PFNGLUNIFORMMATRIX3DVPROC) load(userptr, "glUniformMatrix3dv");
    context->UniformMatrix3x2dv = (PFNGLUNIFORMMATRIX3X2DVPROC) load(userptr, "glUniformMatrix3x2dv");
    context->UniformMatrix3x4dv = (PFNGLUNIFORMMATRIX3X4DVPROC) load(userptr, "glUniformMatrix3x4dv");
    context->UniformMatrix4dv = (PFNGLUNIFORMMATRIX4DVPROC) load(userptr, "glUniformMatrix4dv");
    context->UniformMatrix4x2dv = (PFNGLUNIFORMMATRIX4X2DVPROC) load(userptr, "glUniformMatrix4x2dv");
    context->UniformMatrix4x3dv = (PFNGLUNIFORMMATRIX4X3DVPROC) load(userptr, "glUniformMatrix4x3dv");
    context->UniformSubroutinesuiv = (PFNGLUNIFORMSUBROUTINESUIVPROC) load(userptr, "glUniformSubroutinesuiv");
}
static void glad_gl_load_GL_VERSION_4_1(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_4_1) return;
    context->ActiveShaderProgram = (PFNGLACTIVESHADERPROGRAMPROC) load(userptr, "glActiveShaderProgram");
    context->BindProgramPipeline = (PFNGLBINDPROGRAMPIPELINEPROC) load(userptr, "glBindProgramPipeline");
    context->ClearDepthf = (PFNGLCLEARDEPTHFPROC) load(userptr, "glClearDepthf");
    context->CreateShaderProgramv = (PFNGLCREATESHADERPROGRAMVPROC) load(userptr, "glCreateShaderProgramv");
    context->DeleteProgramPipelines = (PFNGLDELETEPROGRAMPIPELINESPROC) load(userptr, "glDeleteProgramPipelines");
    context->DepthRangeArrayv = (PFNGLDEPTHRANGEARRAYVPROC) load(userptr, "glDepthRangeArrayv");
    context->DepthRangeIndexed = (PFNGLDEPTHRANGEINDEXEDPROC) load(userptr, "glDepthRangeIndexed");
    context->DepthRangef = (PFNGLDEPTHRANGEFPROC) load(userptr, "glDepthRangef");
    context->GenProgramPipelines = (PFNGLGENPROGRAMPIPELINESPROC) load(userptr, "glGenProgramPipelines");
    context->GetDoublei_v = (PFNGLGETDOUBLEI_VPROC) load(userptr, "glGetDoublei_v");
    context->GetFloati_v = (PFNGLGETFLOATI_VPROC) load(userptr, "glGetFloati_v");
    context->GetProgramBinary = (PFNGLGETPROGRAMBINARYPROC) load(userptr, "glGetProgramBinary");
    context->GetProgramPipelineInfoLog = (PFNGLGETPROGRAMPIPELINEINFOLOGPROC) load(userptr, "glGetProgramPipelineInfoLog");
    context->GetProgramPipelineiv = (PFNGLGETPROGRAMPIPELINEIVPROC) load(userptr, "glGetProgramPipelineiv");
    context->GetShaderPrecisionFormat = (PFNGLGETSHADERPRECISIONFORMATPROC) load(userptr, "glGetShaderPrecisionFormat");
    context->GetVertexAttribLdv = (PFNGLGETVERTEXATTRIBLDVPROC) load(userptr, "glGetVertexAttribLdv");
    context->IsProgramPipeline = (PFNGLISPROGRAMPIPELINEPROC) load(userptr, "glIsProgramPipeline");
    context->ProgramBinary = (PFNGLPROGRAMBINARYPROC) load(userptr, "glProgramBinary");
    context->ProgramParameteri = (PFNGLPROGRAMPARAMETERIPROC) load(userptr, "glProgramParameteri");
    context->ProgramUniform1d = (PFNGLPROGRAMUNIFORM1DPROC) load(userptr, "glProgramUniform1d");
    context->ProgramUniform1dv = (PFNGLPROGRAMUNIFORM1DVPROC) load(userptr, "glProgramUniform1dv");
    context->ProgramUniform1f = (PFNGLPROGRAMUNIFORM1FPROC) load(userptr, "glProgramUniform1f");
    context->ProgramUniform1fv = (PFNGLPROGRAMUNIFORM1FVPROC) load(userptr, "glProgramUniform1fv");
    context->ProgramUniform1i = (PFNGLPROGRAMUNIFORM1IPROC) load(userptr, "glProgramUniform1i");
    context->ProgramUniform1iv = (PFNGLPROGRAMUNIFORM1IVPROC) load(userptr, "glProgramUniform1iv");
    context->ProgramUniform1ui = (PFNGLPROGRAMUNIFORM1UIPROC) load(userptr, "glProgramUniform1ui");
    context->ProgramUniform1uiv = (PFNGLPROGRAMUNIFORM1UIVPROC) load(userptr, "glProgramUniform1uiv");
    context->ProgramUniform2d = (PFNGLPROGRAMUNIFORM2DPROC) load(userptr, "glProgramUniform2d");
    context->ProgramUniform2dv = (PFNGLPROGRAMUNIFORM2DVPROC) load(userptr, "glProgramUniform2dv");
    context->ProgramUniform2f = (PFNGLPROGRAMUNIFORM2FPROC) load(userptr, "glProgramUniform2f");
    context->ProgramUniform2fv = (PFNGLPROGRAMUNIFORM2FVPROC) load(userptr, "glProgramUniform2fv");
    context->ProgramUniform2i = (PFNGLPROGRAMUNIFORM2IPROC) load(userptr, "glProgramUniform2i");
    context->ProgramUniform2iv = (PFNGLPROGRAMUNIFORM2IVPROC) load(userptr, "glProgramUniform2iv");
    context->ProgramUniform2ui = (PFNGLPROGRAMUNIFORM2UIPROC) load(userptr, "glProgramUniform2ui");
    context->ProgramUniform2uiv = (PFNGLPROGRAMUNIFORM2UIVPROC) load(userptr, "glProgramUniform2uiv");
    context->ProgramUniform3d = (PFNGLPROGRAMUNIFORM3DPROC) load(userptr, "glProgramUniform3d");
    context->ProgramUniform3dv = (PFNGLPROGRAMUNIFORM3DVPROC) load(userptr, "glProgramUniform3dv");
    context->ProgramUniform3f = (PFNGLPROGRAMUNIFORM3FPROC) load(userptr, "glProgramUniform3f");
    context->ProgramUniform3fv = (PFNGLPROGRAMUNIFORM3FVPROC) load(userptr, "glProgramUniform3fv");
    context->ProgramUniform3i = (PFNGLPROGRAMUNIFORM3IPROC) load(userptr, "glProgramUniform3i");
    context->ProgramUniform3iv = (PFNGLPROGRAMUNIFORM3IVPROC) load(userptr, "glProgramUniform3iv");
    context->ProgramUniform3ui = (PFNGLPROGRAMUNIFORM3UIPROC) load(userptr, "glProgramUniform3ui");
    context->ProgramUniform3uiv = (PFNGLPROGRAMUNIFORM3UIVPROC) load(userptr, "glProgramUniform3uiv");
    context->ProgramUniform4d = (PFNGLPROGRAMUNIFORM4DPROC) load(userptr, "glProgramUniform4d");
    context->ProgramUniform4dv = (PFNGLPROGRAMUNIFORM4DVPROC) load(userptr, "glProgramUniform4dv");
    context->ProgramUniform4f = (PFNGLPROGRAMUNIFORM4FPROC) load(userptr, "glProgramUniform4f");
    context->ProgramUniform4fv = (PFNGLPROGRAMUNIFORM4FVPROC) load(userptr, "glProgramUniform4fv");
    context->ProgramUniform4i = (PFNGLPROGRAMUNIFORM4IPROC) load(userptr, "glProgramUniform4i");
    context->ProgramUniform4iv = (PFNGLPROGRAMUNIFORM4IVPROC) load(userptr, "glProgramUniform4iv");
    context->ProgramUniform4ui = (PFNGLPROGRAMUNIFORM4UIPROC) load(userptr, "glProgramUniform4ui");
    context->ProgramUniform4uiv = (PFNGLPROGRAMUNIFORM4UIVPROC) load(userptr, "glProgramUniform4uiv");
    context->ProgramUniformMatrix2dv = (PFNGLPROGRAMUNIFORMMATRIX2DVPROC) load(userptr, "glProgramUniformMatrix2dv");
    context->ProgramUniformMatrix2fv = (PFNGLPROGRAMUNIFORMMATRIX2FVPROC) load(userptr, "glProgramUniformMatrix2fv");
    context->ProgramUniformMatrix2x3dv = (PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC) load(userptr, "glProgramUniformMatrix2x3dv");
    context->ProgramUniformMatrix2x3fv = (PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC) load(userptr, "glProgramUniformMatrix2x3fv");
    context->ProgramUniformMatrix2x4dv = (PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC) load(userptr, "glProgramUniformMatrix2x4dv");
    context->ProgramUniformMatrix2x4fv = (PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC) load(userptr, "glProgramUniformMatrix2x4fv");
    context->ProgramUniformMatrix3dv = (PFNGLPROGRAMUNIFORMMATRIX3DVPROC) load(userptr, "glProgramUniformMatrix3dv");
    context->ProgramUniformMatrix3fv = (PFNGLPROGRAMUNIFORMMATRIX3FVPROC) load(userptr, "glProgramUniformMatrix3fv");
    context->ProgramUniformMatrix3x2dv = (PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC) load(userptr, "glProgramUniformMatrix3x2dv");
    context->ProgramUniformMatrix3x2fv = (PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC) load(userptr, "glProgramUniformMatrix3x2fv");
    context->ProgramUniformMatrix3x4dv = (PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC) load(userptr, "glProgramUniformMatrix3x4dv");
    context->ProgramUniformMatrix3x4fv = (PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC) load(userptr, "glProgramUniformMatrix3x4fv");
    context->ProgramUniformMatrix4dv = (PFNGLPROGRAMUNIFORMMATRIX4DVPROC) load(userptr, "glProgramUniformMatrix4dv");
    context->ProgramUniformMatrix4fv = (PFNGLPROGRAMUNIFORMMATRIX4FVPROC) load(userptr, "glProgramUniformMatrix4fv");
    context->ProgramUniformMatrix4x2dv = (PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC) load(userptr, "glProgramUniformMatrix4x2dv");
    context->ProgramUniformMatrix4x2fv = (PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC) load(userptr, "glProgramUniformMatrix4x2fv");
    context->ProgramUniformMatrix4x3dv = (PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC) load(userptr, "glProgramUniformMatrix4x3dv");
    context->ProgramUniformMatrix4x3fv = (PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC) load(userptr, "glProgramUniformMatrix4x3fv");
    context->ReleaseShaderCompiler = (PFNGLRELEASESHADERCOMPILERPROC) load(userptr, "glReleaseShaderCompiler");
    context->ScissorArrayv = (PFNGLSCISSORARRAYVPROC) load(userptr, "glScissorArrayv");
    context->ScissorIndexed = (PFNGLSCISSORINDEXEDPROC) load(userptr, "glScissorIndexed");
    context->ScissorIndexedv = (PFNGLSCISSORINDEXEDVPROC) load(userptr, "glScissorIndexedv");
    context->ShaderBinary = (PFNGLSHADERBINARYPROC) load(userptr, "glShaderBinary");
    context->UseProgramStages = (PFNGLUSEPROGRAMSTAGESPROC) load(userptr, "glUseProgramStages");
    context->ValidateProgramPipeline = (PFNGLVALIDATEPROGRAMPIPELINEPROC) load(userptr, "glValidateProgramPipeline");
    context->VertexAttribL1d = (PFNGLVERTEXATTRIBL1DPROC) load(userptr, "glVertexAttribL1d");
    context->VertexAttribL1dv = (PFNGLVERTEXATTRIBL1DVPROC) load(userptr, "glVertexAttribL1dv");
    context->VertexAttribL2d = (PFNGLVERTEXATTRIBL2DPROC) load(userptr, "glVertexAttribL2d");
    context->VertexAttribL2dv = (PFNGLVERTEXATTRIBL2DVPROC) load(userptr, "glVertexAttribL2dv");
    context->VertexAttribL3d = (PFNGLVERTEXATTRIBL3DPROC) load(userptr, "glVertexAttribL3d");
    context->VertexAttribL3dv = (PFNGLVERTEXATTRIBL3DVPROC) load(userptr, "glVertexAttribL3dv");
    context->VertexAttribL4d = (PFNGLVERTEXATTRIBL4DPROC) load(userptr, "glVertexAttribL4d");
    context->VertexAttribL4dv = (PFNGLVERTEXATTRIBL4DVPROC) load(userptr, "glVertexAttribL4dv");
    context->VertexAttribLPointer = (PFNGLVERTEXATTRIBLPOINTERPROC) load(userptr, "glVertexAttribLPointer");
    context->ViewportArrayv = (PFNGLVIEWPORTARRAYVPROC) load(userptr, "glViewportArrayv");
    context->ViewportIndexedf = (PFNGLVIEWPORTINDEXEDFPROC) load(userptr, "glViewportIndexedf");
    context->ViewportIndexedfv = (PFNGLVIEWPORTINDEXEDFVPROC) load(userptr, "glViewportIndexedfv");
}
static void glad_gl_load_GL_VERSION_4_2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_4_2) return;
    context->BindImageTexture = (PFNGLBINDIMAGETEXTUREPROC) load(userptr, "glBindImageTexture");
    context->DrawArraysInstancedBaseInstance = (PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC) load(userptr, "glDrawArraysInstancedBaseInstance");
    context->DrawElementsInstancedBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC) load(userptr, "glDrawElementsInstancedBaseInstance");
    context->DrawElementsInstancedBaseVertexBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC) load(userptr, "glDrawElementsInstancedBaseVertexBaseInstance");
    context->DrawTransformFeedbackInstanced = (PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC) load(userptr, "glDrawTransformFeedbackInstanced");
    context->DrawTransformFeedbackStreamInstanced = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC) load(userptr, "glDrawTransformFeedbackStreamInstanced");
    context->GetActiveAtomicCounterBufferiv = (PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC) load(userptr, "glGetActiveAtomicCounterBufferiv");
    context->GetInternalformativ = (PFNGLGETINTERNALFORMATIVPROC) load(userptr, "glGetInternalformativ");
    context->MemoryBarrier = (PFNGLMEMORYBARRIERPROC) load(userptr, "glMemoryBarrier");
    context->TexStorage1D = (PFNGLTEXSTORAGE1DPROC) load(userptr, "glTexStorage1D");
    context->TexStorage2D = (PFNGLTEXSTORAGE2DPROC) load(userptr, "glTexStorage2D");
    context->TexStorage3D = (PFNGLTEXSTORAGE3DPROC) load(userptr, "glTexStorage3D");
}
static void glad_gl_load_GL_VERSION_4_3(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_4_3) return;
    context->BindVertexBuffer = (PFNGLBINDVERTEXBUFFERPROC) load(userptr, "glBindVertexBuffer");
    context->ClearBufferData = (PFNGLCLEARBUFFERDATAPROC) load(userptr, "glClearBufferData");
    context->ClearBufferSubData = (PFNGLCLEARBUFFERSUBDATAPROC) load(userptr, "glClearBufferSubData");
    context->CopyImageSubData = (PFNGLCOPYIMAGESUBDATAPROC) load(userptr, "glCopyImageSubData");
    context->DebugMessageCallback = (PFNGLDEBUGMESSAGECALLBACKPROC) load(userptr, "glDebugMessageCallback");
    context->DebugMessageControl = (PFNGLDEBUGMESSAGECONTROLPROC) load(userptr, "glDebugMessageControl");
    context->DebugMessageInsert = (PFNGLDEBUGMESSAGEINSERTPROC) load(userptr, "glDebugMessageInsert");
    context->DispatchCompute = (PFNGLDISPATCHCOMPUTEPROC) load(userptr, "glDispatchCompute");
    context->DispatchComputeIndirect = (PFNGLDISPATCHCOMPUTEINDIRECTPROC) load(userptr, "glDispatchComputeIndirect");
    context->FramebufferParameteri = (PFNGLFRAMEBUFFERPARAMETERIPROC) load(userptr, "glFramebufferParameteri");
    context->GetDebugMessageLog = (PFNGLGETDEBUGMESSAGELOGPROC) load(userptr, "glGetDebugMessageLog");
    context->GetFramebufferParameteriv = (PFNGLGETFRAMEBUFFERPARAMETERIVPROC) load(userptr, "glGetFramebufferParameteriv");
    context->GetInternalformati64v = (PFNGLGETINTERNALFORMATI64VPROC) load(userptr, "glGetInternalformati64v");
    context->GetObjectLabel = (PFNGLGETOBJECTLABELPROC) load(userptr, "glGetObjectLabel");
    context->GetObjectPtrLabel = (PFNGLGETOBJECTPTRLABELPROC) load(userptr, "glGetObjectPtrLabel");
    context->GetPointerv = (PFNGLGETPOINTERVPROC) load(userptr, "glGetPointerv");
    context->GetProgramInterfaceiv = (PFNGLGETPROGRAMINTERFACEIVPROC) load(userptr, "glGetProgramInterfaceiv");
    context->GetProgramResourceIndex = (PFNGLGETPROGRAMRESOURCEINDEXPROC) load(userptr, "glGetProgramResourceIndex");
    context->GetProgramResourceLocation = (PFNGLGETPROGRAMRESOURCELOCATIONPROC) load(userptr, "glGetProgramResourceLocation");
    context->GetProgramResourceLocationIndex = (PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC) load(userptr, "glGetProgramResourceLocationIndex");
    context->GetProgramResourceName = (PFNGLGETPROGRAMRESOURCENAMEPROC) load(userptr, "glGetProgramResourceName");
    context->GetProgramResourceiv = (PFNGLGETPROGRAMRESOURCEIVPROC) load(userptr, "glGetProgramResourceiv");
    context->InvalidateBufferData = (PFNGLINVALIDATEBUFFERDATAPROC) load(userptr, "glInvalidateBufferData");
    context->InvalidateBufferSubData = (PFNGLINVALIDATEBUFFERSUBDATAPROC) load(userptr, "glInvalidateBufferSubData");
    context->InvalidateFramebuffer = (PFNGLINVALIDATEFRAMEBUFFERPROC) load(userptr, "glInvalidateFramebuffer");
    context->InvalidateSubFramebuffer = (PFNGLINVALIDATESUBFRAMEBUFFERPROC) load(userptr, "glInvalidateSubFramebuffer");
    context->InvalidateTexImage = (PFNGLINVALIDATETEXIMAGEPROC) load(userptr, "glInvalidateTexImage");
    context->InvalidateTexSubImage = (PFNGLINVALIDATETEXSUBIMAGEPROC) load(userptr, "glInvalidateTexSubImage");
    context->MultiDrawArraysIndirect = (PFNGLMULTIDRAWARRAYSINDIRECTPROC) load(userptr, "glMultiDrawArraysIndirect");
    context->MultiDrawElementsIndirect = (PFNGLMULTIDRAWELEMENTSINDIRECTPROC) load(userptr, "glMultiDrawElementsIndirect");
    context->ObjectLabel = (PFNGLOBJECTLABELPROC) load(userptr, "glObjectLabel");
    context->ObjectPtrLabel = (PFNGLOBJECTPTRLABELPROC) load(userptr, "glObjectPtrLabel");
    context->PopDebugGroup = (PFNGLPOPDEBUGGROUPPROC) load(userptr, "glPopDebugGroup");
    context->PushDebugGroup = (PFNGLPUSHDEBUGGROUPPROC) load(userptr, "glPushDebugGroup");
    context->ShaderStorageBlockBinding = (PFNGLSHADERSTORAGEBLOCKBINDINGPROC) load(userptr, "glShaderStorageBlockBinding");
    context->TexBufferRange = (PFNGLTEXBUFFERRANGEPROC) load(userptr, "glTexBufferRange");
    context->TexStorage2DMultisample = (PFNGLTEXSTORAGE2DMULTISAMPLEPROC) load(userptr, "glTexStorage2DMultisample");
    context->TexStorage3DMultisample = (PFNGLTEXSTORAGE3DMULTISAMPLEPROC) load(userptr, "glTexStorage3DMultisample");
    context->TextureView = (PFNGLTEXTUREVIEWPROC) load(userptr, "glTextureView");
    context->VertexAttribBinding = (PFNGLVERTEXATTRIBBINDINGPROC) load(userptr, "glVertexAttribBinding");
    context->VertexAttribFormat = (PFNGLVERTEXATTRIBFORMATPROC) load(userptr, "glVertexAttribFormat");
    context->VertexAttribIFormat = (PFNGLVERTEXATTRIBIFORMATPROC) load(userptr, "glVertexAttribIFormat");
    context->VertexAttribLFormat = (PFNGLVERTEXATTRIBLFORMATPROC) load(userptr, "glVertexAttribLFormat");
    context->VertexBindingDivisor = (PFNGLVERTEXBINDINGDIVISORPROC) load(userptr, "glVertexBindingDivisor");
}
static void glad_gl_load_GL_VERSION_4_4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_4_4) return;
    context->BindBuffersBase = (PFNGLBINDBUFFERSBASEPROC) load(userptr, "glBindBuffersBase");
    context->BindBuffersRange = (PFNGLBINDBUFFERSRANGEPROC) load(userptr, "glBindBuffersRange");
    context->BindImageTextures = (PFNGLBINDIMAGETEXTURESPROC) load(userptr, "glBindImageTextures");
    context->BindSamplers = (PFNGLBINDSAMPLERSPROC) load(userptr, "glBindSamplers");
    context->BindTextures = (PFNGLBINDTEXTURESPROC) load(userptr, "glBindTextures");
    context->BindVertexBuffers = (PFNGLBINDVERTEXBUFFERSPROC) load(userptr, "glBindVertexBuffers");
    context->BufferStorage = (PFNGLBUFFERSTORAGEPROC) load(userptr, "glBufferStorage");
    context->ClearTexImage = (PFNGLCLEARTEXIMAGEPROC) load(userptr, "glClearTexImage");
    context->ClearTexSubImage = (PFNGLCLEARTEXSUBIMAGEPROC) load(userptr, "glClearTexSubImage");
}
static void glad_gl_load_GL_VERSION_4_5(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_4_5) return;
    context->BindTextureUnit = (PFNGLBINDTEXTUREUNITPROC) load(userptr, "glBindTextureUnit");
    context->BlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC) load(userptr, "glBlitNamedFramebuffer");
    context->CheckNamedFramebufferStatus = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC) load(userptr, "glCheckNamedFramebufferStatus");
    context->ClearNamedBufferData = (PFNGLCLEARNAMEDBUFFERDATAPROC) load(userptr, "glClearNamedBufferData");
    context->ClearNamedBufferSubData = (PFNGLCLEARNAMEDBUFFERSUBDATAPROC) load(userptr, "glClearNamedBufferSubData");
    context->ClearNamedFramebufferfi = (PFNGLCLEARNAMEDFRAMEBUFFERFIPROC) load(userptr, "glClearNamedFramebufferfi");
    context->ClearNamedFramebufferfv = (PFNGLCLEARNAMEDFRAMEBUFFERFVPROC) load(userptr, "glClearNamedFramebufferfv");
    context->ClearNamedFramebufferiv = (PFNGLCLEARNAMEDFRAMEBUFFERIVPROC) load(userptr, "glClearNamedFramebufferiv");
    context->ClearNamedFramebufferuiv = (PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC) load(userptr, "glClearNamedFramebufferuiv");
    context->ClipControl = (PFNGLCLIPCONTROLPROC) load(userptr, "glClipControl");
    context->CompressedTextureSubImage1D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC) load(userptr, "glCompressedTextureSubImage1D");
    context->CompressedTextureSubImage2D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC) load(userptr, "glCompressedTextureSubImage2D");
    context->CompressedTextureSubImage3D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC) load(userptr, "glCompressedTextureSubImage3D");
    context->CopyNamedBufferSubData = (PFNGLCOPYNAMEDBUFFERSUBDATAPROC) load(userptr, "glCopyNamedBufferSubData");
    context->CopyTextureSubImage1D = (PFNGLCOPYTEXTURESUBIMAGE1DPROC) load(userptr, "glCopyTextureSubImage1D");
    context->CopyTextureSubImage2D = (PFNGLCOPYTEXTURESUBIMAGE2DPROC) load(userptr, "glCopyTextureSubImage2D");
    context->CopyTextureSubImage3D = (PFNGLCOPYTEXTURESUBIMAGE3DPROC) load(userptr, "glCopyTextureSubImage3D");
    context->CreateBuffers = (PFNGLCREATEBUFFERSPROC) load(userptr, "glCreateBuffers");
    context->CreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC) load(userptr, "glCreateFramebuffers");
    context->CreateProgramPipelines = (PFNGLCREATEPROGRAMPIPELINESPROC) load(userptr, "glCreateProgramPipelines");
    context->CreateQueries = (PFNGLCREATEQUERIESPROC) load(userptr, "glCreateQueries");
    context->CreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC) load(userptr, "glCreateRenderbuffers");
    context->CreateSamplers = (PFNGLCREATESAMPLERSPROC) load(userptr, "glCreateSamplers");
    context->CreateTextures = (PFNGLCREATETEXTURESPROC) load(userptr, "glCreateTextures");
    context->CreateTransformFeedbacks = (PFNGLCREATETRANSFORMFEEDBACKSPROC) load(userptr, "glCreateTransformFeedbacks");
    context->CreateVertexArrays = (PFNGLCREATEVERTEXARRAYSPROC) load(userptr, "glCreateVertexArrays");
    context->DisableVertexArrayAttrib = (PFNGLDISABLEVERTEXARRAYATTRIBPROC) load(userptr, "glDisableVertexArrayAttrib");
    context->EnableVertexArrayAttrib = (PFNGLENABLEVERTEXARRAYATTRIBPROC) load(userptr, "glEnableVertexArrayAttrib");
    context->FlushMappedNamedBufferRange = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC) load(userptr, "glFlushMappedNamedBufferRange");
    context->GenerateTextureMipmap = (PFNGLGENERATETEXTUREMIPMAPPROC) load(userptr, "glGenerateTextureMipmap");
    context->GetCompressedTextureImage = (PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC) load(userptr, "glGetCompressedTextureImage");
    context->GetCompressedTextureSubImage = (PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC) load(userptr, "glGetCompressedTextureSubImage");
    context->GetGraphicsResetStatus = (PFNGLGETGRAPHICSRESETSTATUSPROC) load(userptr, "glGetGraphicsResetStatus");
    context->GetNamedBufferParameteri64v = (PFNGLGETNAMEDBUFFERPARAMETERI64VPROC) load(userptr, "glGetNamedBufferParameteri64v");
    context->GetNamedBufferParameteriv = (PFNGLGETNAMEDBUFFERPARAMETERIVPROC) load(userptr, "glGetNamedBufferParameteriv");
    context->GetNamedBufferPointerv = (PFNGLGETNAMEDBUFFERPOINTERVPROC) load(userptr, "glGetNamedBufferPointerv");
    context->GetNamedBufferSubData = (PFNGLGETNAMEDBUFFERSUBDATAPROC) load(userptr, "glGetNamedBufferSubData");
    context->GetNamedFramebufferAttachmentParameteriv = (PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVPROC) load(userptr, "glGetNamedFramebufferAttachmentParameteriv");
    context->GetNamedFramebufferParameteriv = (PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVPROC) load(userptr, "glGetNamedFramebufferParameteriv");
    context->GetNamedRenderbufferParameteriv = (PFNGLGETNAMEDRENDERBUFFERPARAMETERIVPROC) load(userptr, "glGetNamedRenderbufferParameteriv");
    context->GetQueryBufferObjecti64v = (PFNGLGETQUERYBUFFEROBJECTI64VPROC) load(userptr, "glGetQueryBufferObjecti64v");
    context->GetQueryBufferObjectiv = (PFNGLGETQUERYBUFFEROBJECTIVPROC) load(userptr, "glGetQueryBufferObjectiv");
    context->GetQueryBufferObjectui64v = (PFNGLGETQUERYBUFFEROBJECTUI64VPROC) load(userptr, "glGetQueryBufferObjectui64v");
    context->GetQueryBufferObjectuiv = (PFNGLGETQUERYBUFFEROBJECTUIVPROC) load(userptr, "glGetQueryBufferObjectuiv");
    context->GetTextureImage = (PFNGLGETTEXTUREIMAGEPROC) load(userptr, "glGetTextureImage");
    context->GetTextureLevelParameterfv = (PFNGLGETTEXTURELEVELPARAMETERFVPROC) load(userptr, "glGetTextureLevelParameterfv");
    context->GetTextureLevelParameteriv = (PFNGLGETTEXTURELEVELPARAMETERIVPROC) load(userptr, "glGetTextureLevelParameteriv");
    context->GetTextureParameterIiv = (PFNGLGETTEXTUREPARAMETERIIVPROC) load(userptr, "glGetTextureParameterIiv");
    context->GetTextureParameterIuiv = (PFNGLGETTEXTUREPARAMETERIUIVPROC) load(userptr, "glGetTextureParameterIuiv");
    context->GetTextureParameterfv = (PFNGLGETTEXTUREPARAMETERFVPROC) load(userptr, "glGetTextureParameterfv");
    context->GetTextureParameteriv = (PFNGLGETTEXTUREPARAMETERIVPROC) load(userptr, "glGetTextureParameteriv");
    context->GetTextureSubImage = (PFNGLGETTEXTURESUBIMAGEPROC) load(userptr, "glGetTextureSubImage");
    context->GetTransformFeedbacki64_v = (PFNGLGETTRANSFORMFEEDBACKI64_VPROC) load(userptr, "glGetTransformFeedbacki64_v");
    context->GetTransformFeedbacki_v = (PFNGLGETTRANSFORMFEEDBACKI_VPROC) load(userptr, "glGetTransformFeedbacki_v");
    context->GetTransformFeedbackiv = (PFNGLGETTRANSFORMFEEDBACKIVPROC) load(userptr, "glGetTransformFeedbackiv");
    context->GetVertexArrayIndexed64iv = (PFNGLGETVERTEXARRAYINDEXED64IVPROC) load(userptr, "glGetVertexArrayIndexed64iv");
    context->GetVertexArrayIndexediv = (PFNGLGETVERTEXARRAYINDEXEDIVPROC) load(userptr, "glGetVertexArrayIndexediv");
    context->GetVertexArrayiv = (PFNGLGETVERTEXARRAYIVPROC) load(userptr, "glGetVertexArrayiv");
    context->GetnColorTable = (PFNGLGETNCOLORTABLEPROC) load(userptr, "glGetnColorTable");
    context->GetnCompressedTexImage = (PFNGLGETNCOMPRESSEDTEXIMAGEPROC) load(userptr, "glGetnCompressedTexImage");
    context->GetnConvolutionFilter = (PFNGLGETNCONVOLUTIONFILTERPROC) load(userptr, "glGetnConvolutionFilter");
    context->GetnHistogram = (PFNGLGETNHISTOGRAMPROC) load(userptr, "glGetnHistogram");
    context->GetnMapdv = (PFNGLGETNMAPDVPROC) load(userptr, "glGetnMapdv");
    context->GetnMapfv = (PFNGLGETNMAPFVPROC) load(userptr, "glGetnMapfv");
    context->GetnMapiv = (PFNGLGETNMAPIVPROC) load(userptr, "glGetnMapiv");
    context->GetnMinmax = (PFNGLGETNMINMAXPROC) load(userptr, "glGetnMinmax");
    context->GetnPixelMapfv = (PFNGLGETNPIXELMAPFVPROC) load(userptr, "glGetnPixelMapfv");
    context->GetnPixelMapuiv = (PFNGLGETNPIXELMAPUIVPROC) load(userptr, "glGetnPixelMapuiv");
    context->GetnPixelMapusv = (PFNGLGETNPIXELMAPUSVPROC) load(userptr, "glGetnPixelMapusv");
    context->GetnPolygonStipple = (PFNGLGETNPOLYGONSTIPPLEPROC) load(userptr, "glGetnPolygonStipple");
    context->GetnSeparableFilter = (PFNGLGETNSEPARABLEFILTERPROC) load(userptr, "glGetnSeparableFilter");
    context->GetnTexImage = (PFNGLGETNTEXIMAGEPROC) load(userptr, "glGetnTexImage");
    context->GetnUniformdv = (PFNGLGETNUNIFORMDVPROC) load(userptr, "glGetnUniformdv");
    context->GetnUniformfv = (PFNGLGETNUNIFORMFVPROC) load(userptr, "glGetnUniformfv");
    context->GetnUniformiv = (PFNGLGETNUNIFORMIVPROC) load(userptr, "glGetnUniformiv");
    context->GetnUniformuiv = (PFNGLGETNUNIFORMUIVPROC) load(userptr, "glGetnUniformuiv");
    context->InvalidateNamedFramebufferData = (PFNGLINVALIDATENAMEDFRAMEBUFFERDATAPROC) load(userptr, "glInvalidateNamedFramebufferData");
    context->InvalidateNamedFramebufferSubData = (PFNGLINVALIDATENAMEDFRAMEBUFFERSUBDATAPROC) load(userptr, "glInvalidateNamedFramebufferSubData");
    context->MapNamedBuffer = (PFNGLMAPNAMEDBUFFERPROC) load(userptr, "glMapNamedBuffer");
    context->MapNamedBufferRange = (PFNGLMAPNAMEDBUFFERRANGEPROC) load(userptr, "glMapNamedBufferRange");
    context->MemoryBarrierByRegion = (PFNGLMEMORYBARRIERBYREGIONPROC) load(userptr, "glMemoryBarrierByRegion");
    context->NamedBufferData = (PFNGLNAMEDBUFFERDATAPROC) load(userptr, "glNamedBufferData");
    context->NamedBufferStorage = (PFNGLNAMEDBUFFERSTORAGEPROC) load(userptr, "glNamedBufferStorage");
    context->NamedBufferSubData = (PFNGLNAMEDBUFFERSUBDATAPROC) load(userptr, "glNamedBufferSubData");
    context->NamedFramebufferDrawBuffer = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC) load(userptr, "glNamedFramebufferDrawBuffer");
    context->NamedFramebufferDrawBuffers = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC) load(userptr, "glNamedFramebufferDrawBuffers");
    context->NamedFramebufferParameteri = (PFNGLNAMEDFRAMEBUFFERPARAMETERIPROC) load(userptr, "glNamedFramebufferParameteri");
    context->NamedFramebufferReadBuffer = (PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC) load(userptr, "glNamedFramebufferReadBuffer");
    context->NamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC) load(userptr, "glNamedFramebufferRenderbuffer");
    context->NamedFramebufferTexture = (PFNGLNAMEDFRAMEBUFFERTEXTUREPROC) load(userptr, "glNamedFramebufferTexture");
    context->NamedFramebufferTextureLayer = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC) load(userptr, "glNamedFramebufferTextureLayer");
    context->NamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC) load(userptr, "glNamedRenderbufferStorage");
    context->NamedRenderbufferStorageMultisample = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEPROC) load(userptr, "glNamedRenderbufferStorageMultisample");
    context->ReadnPixels = (PFNGLREADNPIXELSPROC) load(userptr, "glReadnPixels");
    context->TextureBarrier = (PFNGLTEXTUREBARRIERPROC) load(userptr, "glTextureBarrier");
    context->TextureBuffer = (PFNGLTEXTUREBUFFERPROC) load(userptr, "glTextureBuffer");
    context->TextureBufferRange = (PFNGLTEXTUREBUFFERRANGEPROC) load(userptr, "glTextureBufferRange");
    context->TextureParameterIiv = (PFNGLTEXTUREPARAMETERIIVPROC) load(userptr, "glTextureParameterIiv");
    context->TextureParameterIuiv = (PFNGLTEXTUREPARAMETERIUIVPROC) load(userptr, "glTextureParameterIuiv");
    context->TextureParameterf = (PFNGLTEXTUREPARAMETERFPROC) load(userptr, "glTextureParameterf");
    context->TextureParameterfv = (PFNGLTEXTUREPARAMETERFVPROC) load(userptr, "glTextureParameterfv");
    context->TextureParameteri = (PFNGLTEXTUREPARAMETERIPROC) load(userptr, "glTextureParameteri");
    context->TextureParameteriv = (PFNGLTEXTUREPARAMETERIVPROC) load(userptr, "glTextureParameteriv");
    context->TextureStorage1D = (PFNGLTEXTURESTORAGE1DPROC) load(userptr, "glTextureStorage1D");
    context->TextureStorage2D = (PFNGLTEXTURESTORAGE2DPROC) load(userptr, "glTextureStorage2D");
    context->TextureStorage2DMultisample = (PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC) load(userptr, "glTextureStorage2DMultisample");
    context->TextureStorage3D = (PFNGLTEXTURESTORAGE3DPROC) load(userptr, "glTextureStorage3D");
    context->TextureStorage3DMultisample = (PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC) load(userptr, "glTextureStorage3DMultisample");
    context->TextureSubImage1D = (PFNGLTEXTURESUBIMAGE1DPROC) load(userptr, "glTextureSubImage1D");
    context->TextureSubImage2D = (PFNGLTEXTURESUBIMAGE2DPROC) load(userptr, "glTextureSubImage2D");
    context->TextureSubImage3D = (PFNGLTEXTURESUBIMAGE3DPROC) load(userptr, "glTextureSubImage3D");
    context->TransformFeedbackBufferBase = (PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC) load(userptr, "glTransformFeedbackBufferBase");
    context->TransformFeedbackBufferRange = (PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC) load(userptr, "glTransformFeedbackBufferRange");
    context->UnmapNamedBuffer = (PFNGLUNMAPNAMEDBUFFERPROC) load(userptr, "glUnmapNamedBuffer");
    context->VertexArrayAttribBinding = (PFNGLVERTEXARRAYATTRIBBINDINGPROC) load(userptr, "glVertexArrayAttribBinding");
    context->VertexArrayAttribFormat = (PFNGLVERTEXARRAYATTRIBFORMATPROC) load(userptr, "glVertexArrayAttribFormat");
    context->VertexArrayAttribIFormat = (PFNGLVERTEXARRAYATTRIBIFORMATPROC) load(userptr, "glVertexArrayAttribIFormat");
    context->VertexArrayAttribLFormat = (PFNGLVERTEXARRAYATTRIBLFORMATPROC) load(userptr, "glVertexArrayAttribLFormat");
    context->VertexArrayBindingDivisor = (PFNGLVERTEXARRAYBINDINGDIVISORPROC) load(userptr, "glVertexArrayBindingDivisor");
    context->VertexArrayElementBuffer = (PFNGLVERTEXARRAYELEMENTBUFFERPROC) load(userptr, "glVertexArrayElementBuffer");
    context->VertexArrayVertexBuffer = (PFNGLVERTEXARRAYVERTEXBUFFERPROC) load(userptr, "glVertexArrayVertexBuffer");
    context->VertexArrayVertexBuffers = (PFNGLVERTEXARRAYVERTEXBUFFERSPROC) load(userptr, "glVertexArrayVertexBuffers");
}
static void glad_gl_load_GL_VERSION_4_6(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->VERSION_4_6) return;
    context->MultiDrawArraysIndirectCount = (PFNGLMULTIDRAWARRAYSINDIRECTCOUNTPROC) load(userptr, "glMultiDrawArraysIndirectCount");
    context->MultiDrawElementsIndirectCount = (PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTPROC) load(userptr, "glMultiDrawElementsIndirectCount");
    context->PolygonOffsetClamp = (PFNGLPOLYGONOFFSETCLAMPPROC) load(userptr, "glPolygonOffsetClamp");
    context->SpecializeShader = (PFNGLSPECIALIZESHADERPROC) load(userptr, "glSpecializeShader");
}
static void glad_gl_load_GL_3DFX_tbuffer(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->_3DFX_tbuffer) return;
    context->TbufferMask3DFX = (PFNGLTBUFFERMASK3DFXPROC) load(userptr, "glTbufferMask3DFX");
}
static void glad_gl_load_GL_AMD_debug_output(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_debug_output) return;
    context->DebugMessageCallbackAMD = (PFNGLDEBUGMESSAGECALLBACKAMDPROC) load(userptr, "glDebugMessageCallbackAMD");
    context->DebugMessageEnableAMD = (PFNGLDEBUGMESSAGEENABLEAMDPROC) load(userptr, "glDebugMessageEnableAMD");
    context->DebugMessageInsertAMD = (PFNGLDEBUGMESSAGEINSERTAMDPROC) load(userptr, "glDebugMessageInsertAMD");
    context->GetDebugMessageLogAMD = (PFNGLGETDEBUGMESSAGELOGAMDPROC) load(userptr, "glGetDebugMessageLogAMD");
}
static void glad_gl_load_GL_AMD_draw_buffers_blend(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_draw_buffers_blend) return;
    context->BlendEquationIndexedAMD = (PFNGLBLENDEQUATIONINDEXEDAMDPROC) load(userptr, "glBlendEquationIndexedAMD");
    context->BlendEquationSeparateIndexedAMD = (PFNGLBLENDEQUATIONSEPARATEINDEXEDAMDPROC) load(userptr, "glBlendEquationSeparateIndexedAMD");
    context->BlendFuncIndexedAMD = (PFNGLBLENDFUNCINDEXEDAMDPROC) load(userptr, "glBlendFuncIndexedAMD");
    context->BlendFuncSeparateIndexedAMD = (PFNGLBLENDFUNCSEPARATEINDEXEDAMDPROC) load(userptr, "glBlendFuncSeparateIndexedAMD");
}
static void glad_gl_load_GL_AMD_framebuffer_multisample_advanced(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_framebuffer_multisample_advanced) return;
    context->NamedRenderbufferStorageMultisampleAdvancedAMD = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEADVANCEDAMDPROC) load(userptr, "glNamedRenderbufferStorageMultisampleAdvancedAMD");
    context->RenderbufferStorageMultisampleAdvancedAMD = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEADVANCEDAMDPROC) load(userptr, "glRenderbufferStorageMultisampleAdvancedAMD");
}
static void glad_gl_load_GL_AMD_framebuffer_sample_positions(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_framebuffer_sample_positions) return;
    context->FramebufferSamplePositionsfvAMD = (PFNGLFRAMEBUFFERSAMPLEPOSITIONSFVAMDPROC) load(userptr, "glFramebufferSamplePositionsfvAMD");
    context->GetFramebufferParameterfvAMD = (PFNGLGETFRAMEBUFFERPARAMETERFVAMDPROC) load(userptr, "glGetFramebufferParameterfvAMD");
    context->GetNamedFramebufferParameterfvAMD = (PFNGLGETNAMEDFRAMEBUFFERPARAMETERFVAMDPROC) load(userptr, "glGetNamedFramebufferParameterfvAMD");
    context->NamedFramebufferSamplePositionsfvAMD = (PFNGLNAMEDFRAMEBUFFERSAMPLEPOSITIONSFVAMDPROC) load(userptr, "glNamedFramebufferSamplePositionsfvAMD");
}
static void glad_gl_load_GL_AMD_gpu_shader_int64(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_gpu_shader_int64) return;
    context->GetUniformi64vNV = (PFNGLGETUNIFORMI64VNVPROC) load(userptr, "glGetUniformi64vNV");
    context->GetUniformui64vNV = (PFNGLGETUNIFORMUI64VNVPROC) load(userptr, "glGetUniformui64vNV");
    context->ProgramUniform1i64NV = (PFNGLPROGRAMUNIFORM1I64NVPROC) load(userptr, "glProgramUniform1i64NV");
    context->ProgramUniform1i64vNV = (PFNGLPROGRAMUNIFORM1I64VNVPROC) load(userptr, "glProgramUniform1i64vNV");
    context->ProgramUniform1ui64NV = (PFNGLPROGRAMUNIFORM1UI64NVPROC) load(userptr, "glProgramUniform1ui64NV");
    context->ProgramUniform1ui64vNV = (PFNGLPROGRAMUNIFORM1UI64VNVPROC) load(userptr, "glProgramUniform1ui64vNV");
    context->ProgramUniform2i64NV = (PFNGLPROGRAMUNIFORM2I64NVPROC) load(userptr, "glProgramUniform2i64NV");
    context->ProgramUniform2i64vNV = (PFNGLPROGRAMUNIFORM2I64VNVPROC) load(userptr, "glProgramUniform2i64vNV");
    context->ProgramUniform2ui64NV = (PFNGLPROGRAMUNIFORM2UI64NVPROC) load(userptr, "glProgramUniform2ui64NV");
    context->ProgramUniform2ui64vNV = (PFNGLPROGRAMUNIFORM2UI64VNVPROC) load(userptr, "glProgramUniform2ui64vNV");
    context->ProgramUniform3i64NV = (PFNGLPROGRAMUNIFORM3I64NVPROC) load(userptr, "glProgramUniform3i64NV");
    context->ProgramUniform3i64vNV = (PFNGLPROGRAMUNIFORM3I64VNVPROC) load(userptr, "glProgramUniform3i64vNV");
    context->ProgramUniform3ui64NV = (PFNGLPROGRAMUNIFORM3UI64NVPROC) load(userptr, "glProgramUniform3ui64NV");
    context->ProgramUniform3ui64vNV = (PFNGLPROGRAMUNIFORM3UI64VNVPROC) load(userptr, "glProgramUniform3ui64vNV");
    context->ProgramUniform4i64NV = (PFNGLPROGRAMUNIFORM4I64NVPROC) load(userptr, "glProgramUniform4i64NV");
    context->ProgramUniform4i64vNV = (PFNGLPROGRAMUNIFORM4I64VNVPROC) load(userptr, "glProgramUniform4i64vNV");
    context->ProgramUniform4ui64NV = (PFNGLPROGRAMUNIFORM4UI64NVPROC) load(userptr, "glProgramUniform4ui64NV");
    context->ProgramUniform4ui64vNV = (PFNGLPROGRAMUNIFORM4UI64VNVPROC) load(userptr, "glProgramUniform4ui64vNV");
    context->Uniform1i64NV = (PFNGLUNIFORM1I64NVPROC) load(userptr, "glUniform1i64NV");
    context->Uniform1i64vNV = (PFNGLUNIFORM1I64VNVPROC) load(userptr, "glUniform1i64vNV");
    context->Uniform1ui64NV = (PFNGLUNIFORM1UI64NVPROC) load(userptr, "glUniform1ui64NV");
    context->Uniform1ui64vNV = (PFNGLUNIFORM1UI64VNVPROC) load(userptr, "glUniform1ui64vNV");
    context->Uniform2i64NV = (PFNGLUNIFORM2I64NVPROC) load(userptr, "glUniform2i64NV");
    context->Uniform2i64vNV = (PFNGLUNIFORM2I64VNVPROC) load(userptr, "glUniform2i64vNV");
    context->Uniform2ui64NV = (PFNGLUNIFORM2UI64NVPROC) load(userptr, "glUniform2ui64NV");
    context->Uniform2ui64vNV = (PFNGLUNIFORM2UI64VNVPROC) load(userptr, "glUniform2ui64vNV");
    context->Uniform3i64NV = (PFNGLUNIFORM3I64NVPROC) load(userptr, "glUniform3i64NV");
    context->Uniform3i64vNV = (PFNGLUNIFORM3I64VNVPROC) load(userptr, "glUniform3i64vNV");
    context->Uniform3ui64NV = (PFNGLUNIFORM3UI64NVPROC) load(userptr, "glUniform3ui64NV");
    context->Uniform3ui64vNV = (PFNGLUNIFORM3UI64VNVPROC) load(userptr, "glUniform3ui64vNV");
    context->Uniform4i64NV = (PFNGLUNIFORM4I64NVPROC) load(userptr, "glUniform4i64NV");
    context->Uniform4i64vNV = (PFNGLUNIFORM4I64VNVPROC) load(userptr, "glUniform4i64vNV");
    context->Uniform4ui64NV = (PFNGLUNIFORM4UI64NVPROC) load(userptr, "glUniform4ui64NV");
    context->Uniform4ui64vNV = (PFNGLUNIFORM4UI64VNVPROC) load(userptr, "glUniform4ui64vNV");
}
static void glad_gl_load_GL_AMD_interleaved_elements(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_interleaved_elements) return;
    context->VertexAttribParameteriAMD = (PFNGLVERTEXATTRIBPARAMETERIAMDPROC) load(userptr, "glVertexAttribParameteriAMD");
}
static void glad_gl_load_GL_AMD_multi_draw_indirect(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_multi_draw_indirect) return;
    context->MultiDrawArraysIndirectAMD = (PFNGLMULTIDRAWARRAYSINDIRECTAMDPROC) load(userptr, "glMultiDrawArraysIndirectAMD");
    context->MultiDrawElementsIndirectAMD = (PFNGLMULTIDRAWELEMENTSINDIRECTAMDPROC) load(userptr, "glMultiDrawElementsIndirectAMD");
}
static void glad_gl_load_GL_AMD_name_gen_delete(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_name_gen_delete) return;
    context->DeleteNamesAMD = (PFNGLDELETENAMESAMDPROC) load(userptr, "glDeleteNamesAMD");
    context->GenNamesAMD = (PFNGLGENNAMESAMDPROC) load(userptr, "glGenNamesAMD");
    context->IsNameAMD = (PFNGLISNAMEAMDPROC) load(userptr, "glIsNameAMD");
}
static void glad_gl_load_GL_AMD_occlusion_query_event(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_occlusion_query_event) return;
    context->QueryObjectParameteruiAMD = (PFNGLQUERYOBJECTPARAMETERUIAMDPROC) load(userptr, "glQueryObjectParameteruiAMD");
}
static void glad_gl_load_GL_AMD_performance_monitor(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_performance_monitor) return;
    context->BeginPerfMonitorAMD = (PFNGLBEGINPERFMONITORAMDPROC) load(userptr, "glBeginPerfMonitorAMD");
    context->DeletePerfMonitorsAMD = (PFNGLDELETEPERFMONITORSAMDPROC) load(userptr, "glDeletePerfMonitorsAMD");
    context->EndPerfMonitorAMD = (PFNGLENDPERFMONITORAMDPROC) load(userptr, "glEndPerfMonitorAMD");
    context->GenPerfMonitorsAMD = (PFNGLGENPERFMONITORSAMDPROC) load(userptr, "glGenPerfMonitorsAMD");
    context->GetPerfMonitorCounterDataAMD = (PFNGLGETPERFMONITORCOUNTERDATAAMDPROC) load(userptr, "glGetPerfMonitorCounterDataAMD");
    context->GetPerfMonitorCounterInfoAMD = (PFNGLGETPERFMONITORCOUNTERINFOAMDPROC) load(userptr, "glGetPerfMonitorCounterInfoAMD");
    context->GetPerfMonitorCounterStringAMD = (PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC) load(userptr, "glGetPerfMonitorCounterStringAMD");
    context->GetPerfMonitorCountersAMD = (PFNGLGETPERFMONITORCOUNTERSAMDPROC) load(userptr, "glGetPerfMonitorCountersAMD");
    context->GetPerfMonitorGroupStringAMD = (PFNGLGETPERFMONITORGROUPSTRINGAMDPROC) load(userptr, "glGetPerfMonitorGroupStringAMD");
    context->GetPerfMonitorGroupsAMD = (PFNGLGETPERFMONITORGROUPSAMDPROC) load(userptr, "glGetPerfMonitorGroupsAMD");
    context->SelectPerfMonitorCountersAMD = (PFNGLSELECTPERFMONITORCOUNTERSAMDPROC) load(userptr, "glSelectPerfMonitorCountersAMD");
}
static void glad_gl_load_GL_AMD_sample_positions(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_sample_positions) return;
    context->SetMultisamplefvAMD = (PFNGLSETMULTISAMPLEFVAMDPROC) load(userptr, "glSetMultisamplefvAMD");
}
static void glad_gl_load_GL_AMD_sparse_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_sparse_texture) return;
    context->TexStorageSparseAMD = (PFNGLTEXSTORAGESPARSEAMDPROC) load(userptr, "glTexStorageSparseAMD");
    context->TextureStorageSparseAMD = (PFNGLTEXTURESTORAGESPARSEAMDPROC) load(userptr, "glTextureStorageSparseAMD");
}
static void glad_gl_load_GL_AMD_stencil_operation_extended(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_stencil_operation_extended) return;
    context->StencilOpValueAMD = (PFNGLSTENCILOPVALUEAMDPROC) load(userptr, "glStencilOpValueAMD");
}
static void glad_gl_load_GL_AMD_vertex_shader_tessellator(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->AMD_vertex_shader_tessellator) return;
    context->TessellationFactorAMD = (PFNGLTESSELLATIONFACTORAMDPROC) load(userptr, "glTessellationFactorAMD");
    context->TessellationModeAMD = (PFNGLTESSELLATIONMODEAMDPROC) load(userptr, "glTessellationModeAMD");
}
static void glad_gl_load_GL_APPLE_element_array(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_element_array) return;
    context->DrawElementArrayAPPLE = (PFNGLDRAWELEMENTARRAYAPPLEPROC) load(userptr, "glDrawElementArrayAPPLE");
    context->DrawRangeElementArrayAPPLE = (PFNGLDRAWRANGEELEMENTARRAYAPPLEPROC) load(userptr, "glDrawRangeElementArrayAPPLE");
    context->ElementPointerAPPLE = (PFNGLELEMENTPOINTERAPPLEPROC) load(userptr, "glElementPointerAPPLE");
    context->MultiDrawElementArrayAPPLE = (PFNGLMULTIDRAWELEMENTARRAYAPPLEPROC) load(userptr, "glMultiDrawElementArrayAPPLE");
    context->MultiDrawRangeElementArrayAPPLE = (PFNGLMULTIDRAWRANGEELEMENTARRAYAPPLEPROC) load(userptr, "glMultiDrawRangeElementArrayAPPLE");
}
static void glad_gl_load_GL_APPLE_fence(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_fence) return;
    context->DeleteFencesAPPLE = (PFNGLDELETEFENCESAPPLEPROC) load(userptr, "glDeleteFencesAPPLE");
    context->FinishFenceAPPLE = (PFNGLFINISHFENCEAPPLEPROC) load(userptr, "glFinishFenceAPPLE");
    context->FinishObjectAPPLE = (PFNGLFINISHOBJECTAPPLEPROC) load(userptr, "glFinishObjectAPPLE");
    context->GenFencesAPPLE = (PFNGLGENFENCESAPPLEPROC) load(userptr, "glGenFencesAPPLE");
    context->IsFenceAPPLE = (PFNGLISFENCEAPPLEPROC) load(userptr, "glIsFenceAPPLE");
    context->SetFenceAPPLE = (PFNGLSETFENCEAPPLEPROC) load(userptr, "glSetFenceAPPLE");
    context->TestFenceAPPLE = (PFNGLTESTFENCEAPPLEPROC) load(userptr, "glTestFenceAPPLE");
    context->TestObjectAPPLE = (PFNGLTESTOBJECTAPPLEPROC) load(userptr, "glTestObjectAPPLE");
}
static void glad_gl_load_GL_APPLE_flush_buffer_range(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_flush_buffer_range) return;
    context->BufferParameteriAPPLE = (PFNGLBUFFERPARAMETERIAPPLEPROC) load(userptr, "glBufferParameteriAPPLE");
    context->FlushMappedBufferRangeAPPLE = (PFNGLFLUSHMAPPEDBUFFERRANGEAPPLEPROC) load(userptr, "glFlushMappedBufferRangeAPPLE");
}
static void glad_gl_load_GL_APPLE_object_purgeable(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_object_purgeable) return;
    context->GetObjectParameterivAPPLE = (PFNGLGETOBJECTPARAMETERIVAPPLEPROC) load(userptr, "glGetObjectParameterivAPPLE");
    context->ObjectPurgeableAPPLE = (PFNGLOBJECTPURGEABLEAPPLEPROC) load(userptr, "glObjectPurgeableAPPLE");
    context->ObjectUnpurgeableAPPLE = (PFNGLOBJECTUNPURGEABLEAPPLEPROC) load(userptr, "glObjectUnpurgeableAPPLE");
}
static void glad_gl_load_GL_APPLE_texture_range(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_texture_range) return;
    context->GetTexParameterPointervAPPLE = (PFNGLGETTEXPARAMETERPOINTERVAPPLEPROC) load(userptr, "glGetTexParameterPointervAPPLE");
    context->TextureRangeAPPLE = (PFNGLTEXTURERANGEAPPLEPROC) load(userptr, "glTextureRangeAPPLE");
}
static void glad_gl_load_GL_APPLE_vertex_array_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_vertex_array_object) return;
    context->BindVertexArrayAPPLE = (PFNGLBINDVERTEXARRAYAPPLEPROC) load(userptr, "glBindVertexArrayAPPLE");
    context->DeleteVertexArraysAPPLE = (PFNGLDELETEVERTEXARRAYSAPPLEPROC) load(userptr, "glDeleteVertexArraysAPPLE");
    context->GenVertexArraysAPPLE = (PFNGLGENVERTEXARRAYSAPPLEPROC) load(userptr, "glGenVertexArraysAPPLE");
    context->IsVertexArrayAPPLE = (PFNGLISVERTEXARRAYAPPLEPROC) load(userptr, "glIsVertexArrayAPPLE");
}
static void glad_gl_load_GL_APPLE_vertex_array_range(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_vertex_array_range) return;
    context->FlushVertexArrayRangeAPPLE = (PFNGLFLUSHVERTEXARRAYRANGEAPPLEPROC) load(userptr, "glFlushVertexArrayRangeAPPLE");
    context->VertexArrayParameteriAPPLE = (PFNGLVERTEXARRAYPARAMETERIAPPLEPROC) load(userptr, "glVertexArrayParameteriAPPLE");
    context->VertexArrayRangeAPPLE = (PFNGLVERTEXARRAYRANGEAPPLEPROC) load(userptr, "glVertexArrayRangeAPPLE");
}
static void glad_gl_load_GL_APPLE_vertex_program_evaluators(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->APPLE_vertex_program_evaluators) return;
    context->DisableVertexAttribAPPLE = (PFNGLDISABLEVERTEXATTRIBAPPLEPROC) load(userptr, "glDisableVertexAttribAPPLE");
    context->EnableVertexAttribAPPLE = (PFNGLENABLEVERTEXATTRIBAPPLEPROC) load(userptr, "glEnableVertexAttribAPPLE");
    context->IsVertexAttribEnabledAPPLE = (PFNGLISVERTEXATTRIBENABLEDAPPLEPROC) load(userptr, "glIsVertexAttribEnabledAPPLE");
    context->MapVertexAttrib1dAPPLE = (PFNGLMAPVERTEXATTRIB1DAPPLEPROC) load(userptr, "glMapVertexAttrib1dAPPLE");
    context->MapVertexAttrib1fAPPLE = (PFNGLMAPVERTEXATTRIB1FAPPLEPROC) load(userptr, "glMapVertexAttrib1fAPPLE");
    context->MapVertexAttrib2dAPPLE = (PFNGLMAPVERTEXATTRIB2DAPPLEPROC) load(userptr, "glMapVertexAttrib2dAPPLE");
    context->MapVertexAttrib2fAPPLE = (PFNGLMAPVERTEXATTRIB2FAPPLEPROC) load(userptr, "glMapVertexAttrib2fAPPLE");
}
static void glad_gl_load_GL_ARB_ES2_compatibility(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_ES2_compatibility) return;
    context->ClearDepthf = (PFNGLCLEARDEPTHFPROC) load(userptr, "glClearDepthf");
    context->DepthRangef = (PFNGLDEPTHRANGEFPROC) load(userptr, "glDepthRangef");
    context->GetShaderPrecisionFormat = (PFNGLGETSHADERPRECISIONFORMATPROC) load(userptr, "glGetShaderPrecisionFormat");
    context->ReleaseShaderCompiler = (PFNGLRELEASESHADERCOMPILERPROC) load(userptr, "glReleaseShaderCompiler");
    context->ShaderBinary = (PFNGLSHADERBINARYPROC) load(userptr, "glShaderBinary");
}
static void glad_gl_load_GL_ARB_ES3_1_compatibility(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_ES3_1_compatibility) return;
    context->MemoryBarrierByRegion = (PFNGLMEMORYBARRIERBYREGIONPROC) load(userptr, "glMemoryBarrierByRegion");
}
static void glad_gl_load_GL_ARB_ES3_2_compatibility(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_ES3_2_compatibility) return;
    context->PrimitiveBoundingBoxARB = (PFNGLPRIMITIVEBOUNDINGBOXARBPROC) load(userptr, "glPrimitiveBoundingBoxARB");
}
static void glad_gl_load_GL_ARB_base_instance(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_base_instance) return;
    context->DrawArraysInstancedBaseInstance = (PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC) load(userptr, "glDrawArraysInstancedBaseInstance");
    context->DrawElementsInstancedBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC) load(userptr, "glDrawElementsInstancedBaseInstance");
    context->DrawElementsInstancedBaseVertexBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC) load(userptr, "glDrawElementsInstancedBaseVertexBaseInstance");
}
static void glad_gl_load_GL_ARB_bindless_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_bindless_texture) return;
    context->GetImageHandleARB = (PFNGLGETIMAGEHANDLEARBPROC) load(userptr, "glGetImageHandleARB");
    context->GetTextureHandleARB = (PFNGLGETTEXTUREHANDLEARBPROC) load(userptr, "glGetTextureHandleARB");
    context->GetTextureSamplerHandleARB = (PFNGLGETTEXTURESAMPLERHANDLEARBPROC) load(userptr, "glGetTextureSamplerHandleARB");
    context->GetVertexAttribLui64vARB = (PFNGLGETVERTEXATTRIBLUI64VARBPROC) load(userptr, "glGetVertexAttribLui64vARB");
    context->IsImageHandleResidentARB = (PFNGLISIMAGEHANDLERESIDENTARBPROC) load(userptr, "glIsImageHandleResidentARB");
    context->IsTextureHandleResidentARB = (PFNGLISTEXTUREHANDLERESIDENTARBPROC) load(userptr, "glIsTextureHandleResidentARB");
    context->MakeImageHandleNonResidentARB = (PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC) load(userptr, "glMakeImageHandleNonResidentARB");
    context->MakeImageHandleResidentARB = (PFNGLMAKEIMAGEHANDLERESIDENTARBPROC) load(userptr, "glMakeImageHandleResidentARB");
    context->MakeTextureHandleNonResidentARB = (PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC) load(userptr, "glMakeTextureHandleNonResidentARB");
    context->MakeTextureHandleResidentARB = (PFNGLMAKETEXTUREHANDLERESIDENTARBPROC) load(userptr, "glMakeTextureHandleResidentARB");
    context->ProgramUniformHandleui64ARB = (PFNGLPROGRAMUNIFORMHANDLEUI64ARBPROC) load(userptr, "glProgramUniformHandleui64ARB");
    context->ProgramUniformHandleui64vARB = (PFNGLPROGRAMUNIFORMHANDLEUI64VARBPROC) load(userptr, "glProgramUniformHandleui64vARB");
    context->UniformHandleui64ARB = (PFNGLUNIFORMHANDLEUI64ARBPROC) load(userptr, "glUniformHandleui64ARB");
    context->UniformHandleui64vARB = (PFNGLUNIFORMHANDLEUI64VARBPROC) load(userptr, "glUniformHandleui64vARB");
    context->VertexAttribL1ui64ARB = (PFNGLVERTEXATTRIBL1UI64ARBPROC) load(userptr, "glVertexAttribL1ui64ARB");
    context->VertexAttribL1ui64vARB = (PFNGLVERTEXATTRIBL1UI64VARBPROC) load(userptr, "glVertexAttribL1ui64vARB");
}
static void glad_gl_load_GL_ARB_blend_func_extended(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_blend_func_extended) return;
    context->BindFragDataLocationIndexed = (PFNGLBINDFRAGDATALOCATIONINDEXEDPROC) load(userptr, "glBindFragDataLocationIndexed");
    context->GetFragDataIndex = (PFNGLGETFRAGDATAINDEXPROC) load(userptr, "glGetFragDataIndex");
}
static void glad_gl_load_GL_ARB_buffer_storage(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_buffer_storage) return;
    context->BufferStorage = (PFNGLBUFFERSTORAGEPROC) load(userptr, "glBufferStorage");
}
static void glad_gl_load_GL_ARB_cl_event(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_cl_event) return;
    context->CreateSyncFromCLeventARB = (PFNGLCREATESYNCFROMCLEVENTARBPROC) load(userptr, "glCreateSyncFromCLeventARB");
}
static void glad_gl_load_GL_ARB_clear_buffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_clear_buffer_object) return;
    context->ClearBufferData = (PFNGLCLEARBUFFERDATAPROC) load(userptr, "glClearBufferData");
    context->ClearBufferSubData = (PFNGLCLEARBUFFERSUBDATAPROC) load(userptr, "glClearBufferSubData");
}
static void glad_gl_load_GL_ARB_clear_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_clear_texture) return;
    context->ClearTexImage = (PFNGLCLEARTEXIMAGEPROC) load(userptr, "glClearTexImage");
    context->ClearTexSubImage = (PFNGLCLEARTEXSUBIMAGEPROC) load(userptr, "glClearTexSubImage");
}
static void glad_gl_load_GL_ARB_clip_control(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_clip_control) return;
    context->ClipControl = (PFNGLCLIPCONTROLPROC) load(userptr, "glClipControl");
}
static void glad_gl_load_GL_ARB_color_buffer_float(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_color_buffer_float) return;
    context->ClampColorARB = (PFNGLCLAMPCOLORARBPROC) load(userptr, "glClampColorARB");
}
static void glad_gl_load_GL_ARB_compute_shader(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_compute_shader) return;
    context->DispatchCompute = (PFNGLDISPATCHCOMPUTEPROC) load(userptr, "glDispatchCompute");
    context->DispatchComputeIndirect = (PFNGLDISPATCHCOMPUTEINDIRECTPROC) load(userptr, "glDispatchComputeIndirect");
}
static void glad_gl_load_GL_ARB_compute_variable_group_size(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_compute_variable_group_size) return;
    context->DispatchComputeGroupSizeARB = (PFNGLDISPATCHCOMPUTEGROUPSIZEARBPROC) load(userptr, "glDispatchComputeGroupSizeARB");
}
static void glad_gl_load_GL_ARB_copy_buffer(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_copy_buffer) return;
    context->CopyBufferSubData = (PFNGLCOPYBUFFERSUBDATAPROC) load(userptr, "glCopyBufferSubData");
}
static void glad_gl_load_GL_ARB_copy_image(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_copy_image) return;
    context->CopyImageSubData = (PFNGLCOPYIMAGESUBDATAPROC) load(userptr, "glCopyImageSubData");
}
static void glad_gl_load_GL_ARB_debug_output(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_debug_output) return;
    context->DebugMessageCallbackARB = (PFNGLDEBUGMESSAGECALLBACKARBPROC) load(userptr, "glDebugMessageCallbackARB");
    context->DebugMessageControlARB = (PFNGLDEBUGMESSAGECONTROLARBPROC) load(userptr, "glDebugMessageControlARB");
    context->DebugMessageInsertARB = (PFNGLDEBUGMESSAGEINSERTARBPROC) load(userptr, "glDebugMessageInsertARB");
    context->GetDebugMessageLogARB = (PFNGLGETDEBUGMESSAGELOGARBPROC) load(userptr, "glGetDebugMessageLogARB");
}
static void glad_gl_load_GL_ARB_direct_state_access(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_direct_state_access) return;
    context->BindTextureUnit = (PFNGLBINDTEXTUREUNITPROC) load(userptr, "glBindTextureUnit");
    context->BlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC) load(userptr, "glBlitNamedFramebuffer");
    context->CheckNamedFramebufferStatus = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC) load(userptr, "glCheckNamedFramebufferStatus");
    context->ClearNamedBufferData = (PFNGLCLEARNAMEDBUFFERDATAPROC) load(userptr, "glClearNamedBufferData");
    context->ClearNamedBufferSubData = (PFNGLCLEARNAMEDBUFFERSUBDATAPROC) load(userptr, "glClearNamedBufferSubData");
    context->ClearNamedFramebufferfi = (PFNGLCLEARNAMEDFRAMEBUFFERFIPROC) load(userptr, "glClearNamedFramebufferfi");
    context->ClearNamedFramebufferfv = (PFNGLCLEARNAMEDFRAMEBUFFERFVPROC) load(userptr, "glClearNamedFramebufferfv");
    context->ClearNamedFramebufferiv = (PFNGLCLEARNAMEDFRAMEBUFFERIVPROC) load(userptr, "glClearNamedFramebufferiv");
    context->ClearNamedFramebufferuiv = (PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC) load(userptr, "glClearNamedFramebufferuiv");
    context->CompressedTextureSubImage1D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC) load(userptr, "glCompressedTextureSubImage1D");
    context->CompressedTextureSubImage2D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC) load(userptr, "glCompressedTextureSubImage2D");
    context->CompressedTextureSubImage3D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC) load(userptr, "glCompressedTextureSubImage3D");
    context->CopyNamedBufferSubData = (PFNGLCOPYNAMEDBUFFERSUBDATAPROC) load(userptr, "glCopyNamedBufferSubData");
    context->CopyTextureSubImage1D = (PFNGLCOPYTEXTURESUBIMAGE1DPROC) load(userptr, "glCopyTextureSubImage1D");
    context->CopyTextureSubImage2D = (PFNGLCOPYTEXTURESUBIMAGE2DPROC) load(userptr, "glCopyTextureSubImage2D");
    context->CopyTextureSubImage3D = (PFNGLCOPYTEXTURESUBIMAGE3DPROC) load(userptr, "glCopyTextureSubImage3D");
    context->CreateBuffers = (PFNGLCREATEBUFFERSPROC) load(userptr, "glCreateBuffers");
    context->CreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC) load(userptr, "glCreateFramebuffers");
    context->CreateProgramPipelines = (PFNGLCREATEPROGRAMPIPELINESPROC) load(userptr, "glCreateProgramPipelines");
    context->CreateQueries = (PFNGLCREATEQUERIESPROC) load(userptr, "glCreateQueries");
    context->CreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC) load(userptr, "glCreateRenderbuffers");
    context->CreateSamplers = (PFNGLCREATESAMPLERSPROC) load(userptr, "glCreateSamplers");
    context->CreateTextures = (PFNGLCREATETEXTURESPROC) load(userptr, "glCreateTextures");
    context->CreateTransformFeedbacks = (PFNGLCREATETRANSFORMFEEDBACKSPROC) load(userptr, "glCreateTransformFeedbacks");
    context->CreateVertexArrays = (PFNGLCREATEVERTEXARRAYSPROC) load(userptr, "glCreateVertexArrays");
    context->DisableVertexArrayAttrib = (PFNGLDISABLEVERTEXARRAYATTRIBPROC) load(userptr, "glDisableVertexArrayAttrib");
    context->EnableVertexArrayAttrib = (PFNGLENABLEVERTEXARRAYATTRIBPROC) load(userptr, "glEnableVertexArrayAttrib");
    context->FlushMappedNamedBufferRange = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC) load(userptr, "glFlushMappedNamedBufferRange");
    context->GenerateTextureMipmap = (PFNGLGENERATETEXTUREMIPMAPPROC) load(userptr, "glGenerateTextureMipmap");
    context->GetCompressedTextureImage = (PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC) load(userptr, "glGetCompressedTextureImage");
    context->GetNamedBufferParameteri64v = (PFNGLGETNAMEDBUFFERPARAMETERI64VPROC) load(userptr, "glGetNamedBufferParameteri64v");
    context->GetNamedBufferParameteriv = (PFNGLGETNAMEDBUFFERPARAMETERIVPROC) load(userptr, "glGetNamedBufferParameteriv");
    context->GetNamedBufferPointerv = (PFNGLGETNAMEDBUFFERPOINTERVPROC) load(userptr, "glGetNamedBufferPointerv");
    context->GetNamedBufferSubData = (PFNGLGETNAMEDBUFFERSUBDATAPROC) load(userptr, "glGetNamedBufferSubData");
    context->GetNamedFramebufferAttachmentParameteriv = (PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVPROC) load(userptr, "glGetNamedFramebufferAttachmentParameteriv");
    context->GetNamedFramebufferParameteriv = (PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVPROC) load(userptr, "glGetNamedFramebufferParameteriv");
    context->GetNamedRenderbufferParameteriv = (PFNGLGETNAMEDRENDERBUFFERPARAMETERIVPROC) load(userptr, "glGetNamedRenderbufferParameteriv");
    context->GetQueryBufferObjecti64v = (PFNGLGETQUERYBUFFEROBJECTI64VPROC) load(userptr, "glGetQueryBufferObjecti64v");
    context->GetQueryBufferObjectiv = (PFNGLGETQUERYBUFFEROBJECTIVPROC) load(userptr, "glGetQueryBufferObjectiv");
    context->GetQueryBufferObjectui64v = (PFNGLGETQUERYBUFFEROBJECTUI64VPROC) load(userptr, "glGetQueryBufferObjectui64v");
    context->GetQueryBufferObjectuiv = (PFNGLGETQUERYBUFFEROBJECTUIVPROC) load(userptr, "glGetQueryBufferObjectuiv");
    context->GetTextureImage = (PFNGLGETTEXTUREIMAGEPROC) load(userptr, "glGetTextureImage");
    context->GetTextureLevelParameterfv = (PFNGLGETTEXTURELEVELPARAMETERFVPROC) load(userptr, "glGetTextureLevelParameterfv");
    context->GetTextureLevelParameteriv = (PFNGLGETTEXTURELEVELPARAMETERIVPROC) load(userptr, "glGetTextureLevelParameteriv");
    context->GetTextureParameterIiv = (PFNGLGETTEXTUREPARAMETERIIVPROC) load(userptr, "glGetTextureParameterIiv");
    context->GetTextureParameterIuiv = (PFNGLGETTEXTUREPARAMETERIUIVPROC) load(userptr, "glGetTextureParameterIuiv");
    context->GetTextureParameterfv = (PFNGLGETTEXTUREPARAMETERFVPROC) load(userptr, "glGetTextureParameterfv");
    context->GetTextureParameteriv = (PFNGLGETTEXTUREPARAMETERIVPROC) load(userptr, "glGetTextureParameteriv");
    context->GetTransformFeedbacki64_v = (PFNGLGETTRANSFORMFEEDBACKI64_VPROC) load(userptr, "glGetTransformFeedbacki64_v");
    context->GetTransformFeedbacki_v = (PFNGLGETTRANSFORMFEEDBACKI_VPROC) load(userptr, "glGetTransformFeedbacki_v");
    context->GetTransformFeedbackiv = (PFNGLGETTRANSFORMFEEDBACKIVPROC) load(userptr, "glGetTransformFeedbackiv");
    context->GetVertexArrayIndexed64iv = (PFNGLGETVERTEXARRAYINDEXED64IVPROC) load(userptr, "glGetVertexArrayIndexed64iv");
    context->GetVertexArrayIndexediv = (PFNGLGETVERTEXARRAYINDEXEDIVPROC) load(userptr, "glGetVertexArrayIndexediv");
    context->GetVertexArrayiv = (PFNGLGETVERTEXARRAYIVPROC) load(userptr, "glGetVertexArrayiv");
    context->InvalidateNamedFramebufferData = (PFNGLINVALIDATENAMEDFRAMEBUFFERDATAPROC) load(userptr, "glInvalidateNamedFramebufferData");
    context->InvalidateNamedFramebufferSubData = (PFNGLINVALIDATENAMEDFRAMEBUFFERSUBDATAPROC) load(userptr, "glInvalidateNamedFramebufferSubData");
    context->MapNamedBuffer = (PFNGLMAPNAMEDBUFFERPROC) load(userptr, "glMapNamedBuffer");
    context->MapNamedBufferRange = (PFNGLMAPNAMEDBUFFERRANGEPROC) load(userptr, "glMapNamedBufferRange");
    context->NamedBufferData = (PFNGLNAMEDBUFFERDATAPROC) load(userptr, "glNamedBufferData");
    context->NamedBufferStorage = (PFNGLNAMEDBUFFERSTORAGEPROC) load(userptr, "glNamedBufferStorage");
    context->NamedBufferSubData = (PFNGLNAMEDBUFFERSUBDATAPROC) load(userptr, "glNamedBufferSubData");
    context->NamedFramebufferDrawBuffer = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC) load(userptr, "glNamedFramebufferDrawBuffer");
    context->NamedFramebufferDrawBuffers = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC) load(userptr, "glNamedFramebufferDrawBuffers");
    context->NamedFramebufferParameteri = (PFNGLNAMEDFRAMEBUFFERPARAMETERIPROC) load(userptr, "glNamedFramebufferParameteri");
    context->NamedFramebufferReadBuffer = (PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC) load(userptr, "glNamedFramebufferReadBuffer");
    context->NamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC) load(userptr, "glNamedFramebufferRenderbuffer");
    context->NamedFramebufferTexture = (PFNGLNAMEDFRAMEBUFFERTEXTUREPROC) load(userptr, "glNamedFramebufferTexture");
    context->NamedFramebufferTextureLayer = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC) load(userptr, "glNamedFramebufferTextureLayer");
    context->NamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC) load(userptr, "glNamedRenderbufferStorage");
    context->NamedRenderbufferStorageMultisample = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEPROC) load(userptr, "glNamedRenderbufferStorageMultisample");
    context->TextureBuffer = (PFNGLTEXTUREBUFFERPROC) load(userptr, "glTextureBuffer");
    context->TextureBufferRange = (PFNGLTEXTUREBUFFERRANGEPROC) load(userptr, "glTextureBufferRange");
    context->TextureParameterIiv = (PFNGLTEXTUREPARAMETERIIVPROC) load(userptr, "glTextureParameterIiv");
    context->TextureParameterIuiv = (PFNGLTEXTUREPARAMETERIUIVPROC) load(userptr, "glTextureParameterIuiv");
    context->TextureParameterf = (PFNGLTEXTUREPARAMETERFPROC) load(userptr, "glTextureParameterf");
    context->TextureParameterfv = (PFNGLTEXTUREPARAMETERFVPROC) load(userptr, "glTextureParameterfv");
    context->TextureParameteri = (PFNGLTEXTUREPARAMETERIPROC) load(userptr, "glTextureParameteri");
    context->TextureParameteriv = (PFNGLTEXTUREPARAMETERIVPROC) load(userptr, "glTextureParameteriv");
    context->TextureStorage1D = (PFNGLTEXTURESTORAGE1DPROC) load(userptr, "glTextureStorage1D");
    context->TextureStorage2D = (PFNGLTEXTURESTORAGE2DPROC) load(userptr, "glTextureStorage2D");
    context->TextureStorage2DMultisample = (PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC) load(userptr, "glTextureStorage2DMultisample");
    context->TextureStorage3D = (PFNGLTEXTURESTORAGE3DPROC) load(userptr, "glTextureStorage3D");
    context->TextureStorage3DMultisample = (PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC) load(userptr, "glTextureStorage3DMultisample");
    context->TextureSubImage1D = (PFNGLTEXTURESUBIMAGE1DPROC) load(userptr, "glTextureSubImage1D");
    context->TextureSubImage2D = (PFNGLTEXTURESUBIMAGE2DPROC) load(userptr, "glTextureSubImage2D");
    context->TextureSubImage3D = (PFNGLTEXTURESUBIMAGE3DPROC) load(userptr, "glTextureSubImage3D");
    context->TransformFeedbackBufferBase = (PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC) load(userptr, "glTransformFeedbackBufferBase");
    context->TransformFeedbackBufferRange = (PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC) load(userptr, "glTransformFeedbackBufferRange");
    context->UnmapNamedBuffer = (PFNGLUNMAPNAMEDBUFFERPROC) load(userptr, "glUnmapNamedBuffer");
    context->VertexArrayAttribBinding = (PFNGLVERTEXARRAYATTRIBBINDINGPROC) load(userptr, "glVertexArrayAttribBinding");
    context->VertexArrayAttribFormat = (PFNGLVERTEXARRAYATTRIBFORMATPROC) load(userptr, "glVertexArrayAttribFormat");
    context->VertexArrayAttribIFormat = (PFNGLVERTEXARRAYATTRIBIFORMATPROC) load(userptr, "glVertexArrayAttribIFormat");
    context->VertexArrayAttribLFormat = (PFNGLVERTEXARRAYATTRIBLFORMATPROC) load(userptr, "glVertexArrayAttribLFormat");
    context->VertexArrayBindingDivisor = (PFNGLVERTEXARRAYBINDINGDIVISORPROC) load(userptr, "glVertexArrayBindingDivisor");
    context->VertexArrayElementBuffer = (PFNGLVERTEXARRAYELEMENTBUFFERPROC) load(userptr, "glVertexArrayElementBuffer");
    context->VertexArrayVertexBuffer = (PFNGLVERTEXARRAYVERTEXBUFFERPROC) load(userptr, "glVertexArrayVertexBuffer");
    context->VertexArrayVertexBuffers = (PFNGLVERTEXARRAYVERTEXBUFFERSPROC) load(userptr, "glVertexArrayVertexBuffers");
}
static void glad_gl_load_GL_ARB_draw_buffers(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_draw_buffers) return;
    context->DrawBuffersARB = (PFNGLDRAWBUFFERSARBPROC) load(userptr, "glDrawBuffersARB");
}
static void glad_gl_load_GL_ARB_draw_buffers_blend(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_draw_buffers_blend) return;
    context->BlendEquationSeparateiARB = (PFNGLBLENDEQUATIONSEPARATEIARBPROC) load(userptr, "glBlendEquationSeparateiARB");
    context->BlendEquationiARB = (PFNGLBLENDEQUATIONIARBPROC) load(userptr, "glBlendEquationiARB");
    context->BlendFuncSeparateiARB = (PFNGLBLENDFUNCSEPARATEIARBPROC) load(userptr, "glBlendFuncSeparateiARB");
    context->BlendFunciARB = (PFNGLBLENDFUNCIARBPROC) load(userptr, "glBlendFunciARB");
}
static void glad_gl_load_GL_ARB_draw_elements_base_vertex(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_draw_elements_base_vertex) return;
    context->DrawElementsBaseVertex = (PFNGLDRAWELEMENTSBASEVERTEXPROC) load(userptr, "glDrawElementsBaseVertex");
    context->DrawElementsInstancedBaseVertex = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC) load(userptr, "glDrawElementsInstancedBaseVertex");
    context->DrawRangeElementsBaseVertex = (PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC) load(userptr, "glDrawRangeElementsBaseVertex");
    context->MultiDrawElementsBaseVertex = (PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC) load(userptr, "glMultiDrawElementsBaseVertex");
}
static void glad_gl_load_GL_ARB_draw_indirect(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_draw_indirect) return;
    context->DrawArraysIndirect = (PFNGLDRAWARRAYSINDIRECTPROC) load(userptr, "glDrawArraysIndirect");
    context->DrawElementsIndirect = (PFNGLDRAWELEMENTSINDIRECTPROC) load(userptr, "glDrawElementsIndirect");
}
static void glad_gl_load_GL_ARB_draw_instanced(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_draw_instanced) return;
    context->DrawArraysInstancedARB = (PFNGLDRAWARRAYSINSTANCEDARBPROC) load(userptr, "glDrawArraysInstancedARB");
    context->DrawElementsInstancedARB = (PFNGLDRAWELEMENTSINSTANCEDARBPROC) load(userptr, "glDrawElementsInstancedARB");
}
static void glad_gl_load_GL_ARB_fragment_program(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_fragment_program) return;
    context->BindProgramARB = (PFNGLBINDPROGRAMARBPROC) load(userptr, "glBindProgramARB");
    context->DeleteProgramsARB = (PFNGLDELETEPROGRAMSARBPROC) load(userptr, "glDeleteProgramsARB");
    context->GenProgramsARB = (PFNGLGENPROGRAMSARBPROC) load(userptr, "glGenProgramsARB");
    context->GetProgramEnvParameterdvARB = (PFNGLGETPROGRAMENVPARAMETERDVARBPROC) load(userptr, "glGetProgramEnvParameterdvARB");
    context->GetProgramEnvParameterfvARB = (PFNGLGETPROGRAMENVPARAMETERFVARBPROC) load(userptr, "glGetProgramEnvParameterfvARB");
    context->GetProgramLocalParameterdvARB = (PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC) load(userptr, "glGetProgramLocalParameterdvARB");
    context->GetProgramLocalParameterfvARB = (PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC) load(userptr, "glGetProgramLocalParameterfvARB");
    context->GetProgramStringARB = (PFNGLGETPROGRAMSTRINGARBPROC) load(userptr, "glGetProgramStringARB");
    context->GetProgramivARB = (PFNGLGETPROGRAMIVARBPROC) load(userptr, "glGetProgramivARB");
    context->IsProgramARB = (PFNGLISPROGRAMARBPROC) load(userptr, "glIsProgramARB");
    context->ProgramEnvParameter4dARB = (PFNGLPROGRAMENVPARAMETER4DARBPROC) load(userptr, "glProgramEnvParameter4dARB");
    context->ProgramEnvParameter4dvARB = (PFNGLPROGRAMENVPARAMETER4DVARBPROC) load(userptr, "glProgramEnvParameter4dvARB");
    context->ProgramEnvParameter4fARB = (PFNGLPROGRAMENVPARAMETER4FARBPROC) load(userptr, "glProgramEnvParameter4fARB");
    context->ProgramEnvParameter4fvARB = (PFNGLPROGRAMENVPARAMETER4FVARBPROC) load(userptr, "glProgramEnvParameter4fvARB");
    context->ProgramLocalParameter4dARB = (PFNGLPROGRAMLOCALPARAMETER4DARBPROC) load(userptr, "glProgramLocalParameter4dARB");
    context->ProgramLocalParameter4dvARB = (PFNGLPROGRAMLOCALPARAMETER4DVARBPROC) load(userptr, "glProgramLocalParameter4dvARB");
    context->ProgramLocalParameter4fARB = (PFNGLPROGRAMLOCALPARAMETER4FARBPROC) load(userptr, "glProgramLocalParameter4fARB");
    context->ProgramLocalParameter4fvARB = (PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) load(userptr, "glProgramLocalParameter4fvARB");
    context->ProgramStringARB = (PFNGLPROGRAMSTRINGARBPROC) load(userptr, "glProgramStringARB");
}
static void glad_gl_load_GL_ARB_framebuffer_no_attachments(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_framebuffer_no_attachments) return;
    context->FramebufferParameteri = (PFNGLFRAMEBUFFERPARAMETERIPROC) load(userptr, "glFramebufferParameteri");
    context->GetFramebufferParameteriv = (PFNGLGETFRAMEBUFFERPARAMETERIVPROC) load(userptr, "glGetFramebufferParameteriv");
}
static void glad_gl_load_GL_ARB_framebuffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_framebuffer_object) return;
    context->BindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC) load(userptr, "glBindFramebuffer");
    context->BindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC) load(userptr, "glBindRenderbuffer");
    context->BlitFramebuffer = (PFNGLBLITFRAMEBUFFERPROC) load(userptr, "glBlitFramebuffer");
    context->CheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC) load(userptr, "glCheckFramebufferStatus");
    context->DeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC) load(userptr, "glDeleteFramebuffers");
    context->DeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC) load(userptr, "glDeleteRenderbuffers");
    context->FramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC) load(userptr, "glFramebufferRenderbuffer");
    context->FramebufferTexture1D = (PFNGLFRAMEBUFFERTEXTURE1DPROC) load(userptr, "glFramebufferTexture1D");
    context->FramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC) load(userptr, "glFramebufferTexture2D");
    context->FramebufferTexture3D = (PFNGLFRAMEBUFFERTEXTURE3DPROC) load(userptr, "glFramebufferTexture3D");
    context->FramebufferTextureLayer = (PFNGLFRAMEBUFFERTEXTURELAYERPROC) load(userptr, "glFramebufferTextureLayer");
    context->GenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC) load(userptr, "glGenFramebuffers");
    context->GenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC) load(userptr, "glGenRenderbuffers");
    context->GenerateMipmap = (PFNGLGENERATEMIPMAPPROC) load(userptr, "glGenerateMipmap");
    context->GetFramebufferAttachmentParameteriv = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC) load(userptr, "glGetFramebufferAttachmentParameteriv");
    context->GetRenderbufferParameteriv = (PFNGLGETRENDERBUFFERPARAMETERIVPROC) load(userptr, "glGetRenderbufferParameteriv");
    context->IsFramebuffer = (PFNGLISFRAMEBUFFERPROC) load(userptr, "glIsFramebuffer");
    context->IsRenderbuffer = (PFNGLISRENDERBUFFERPROC) load(userptr, "glIsRenderbuffer");
    context->RenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC) load(userptr, "glRenderbufferStorage");
    context->RenderbufferStorageMultisample = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC) load(userptr, "glRenderbufferStorageMultisample");
}
static void glad_gl_load_GL_ARB_geometry_shader4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_geometry_shader4) return;
    context->FramebufferTextureARB = (PFNGLFRAMEBUFFERTEXTUREARBPROC) load(userptr, "glFramebufferTextureARB");
    context->FramebufferTextureFaceARB = (PFNGLFRAMEBUFFERTEXTUREFACEARBPROC) load(userptr, "glFramebufferTextureFaceARB");
    context->FramebufferTextureLayerARB = (PFNGLFRAMEBUFFERTEXTURELAYERARBPROC) load(userptr, "glFramebufferTextureLayerARB");
    context->ProgramParameteriARB = (PFNGLPROGRAMPARAMETERIARBPROC) load(userptr, "glProgramParameteriARB");
}
static void glad_gl_load_GL_ARB_get_program_binary(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_get_program_binary) return;
    context->GetProgramBinary = (PFNGLGETPROGRAMBINARYPROC) load(userptr, "glGetProgramBinary");
    context->ProgramBinary = (PFNGLPROGRAMBINARYPROC) load(userptr, "glProgramBinary");
    context->ProgramParameteri = (PFNGLPROGRAMPARAMETERIPROC) load(userptr, "glProgramParameteri");
}
static void glad_gl_load_GL_ARB_get_texture_sub_image(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_get_texture_sub_image) return;
    context->GetCompressedTextureSubImage = (PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC) load(userptr, "glGetCompressedTextureSubImage");
    context->GetTextureSubImage = (PFNGLGETTEXTURESUBIMAGEPROC) load(userptr, "glGetTextureSubImage");
}
static void glad_gl_load_GL_ARB_gl_spirv(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_gl_spirv) return;
    context->SpecializeShaderARB = (PFNGLSPECIALIZESHADERARBPROC) load(userptr, "glSpecializeShaderARB");
}
static void glad_gl_load_GL_ARB_gpu_shader_fp64(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_gpu_shader_fp64) return;
    context->GetUniformdv = (PFNGLGETUNIFORMDVPROC) load(userptr, "glGetUniformdv");
    context->Uniform1d = (PFNGLUNIFORM1DPROC) load(userptr, "glUniform1d");
    context->Uniform1dv = (PFNGLUNIFORM1DVPROC) load(userptr, "glUniform1dv");
    context->Uniform2d = (PFNGLUNIFORM2DPROC) load(userptr, "glUniform2d");
    context->Uniform2dv = (PFNGLUNIFORM2DVPROC) load(userptr, "glUniform2dv");
    context->Uniform3d = (PFNGLUNIFORM3DPROC) load(userptr, "glUniform3d");
    context->Uniform3dv = (PFNGLUNIFORM3DVPROC) load(userptr, "glUniform3dv");
    context->Uniform4d = (PFNGLUNIFORM4DPROC) load(userptr, "glUniform4d");
    context->Uniform4dv = (PFNGLUNIFORM4DVPROC) load(userptr, "glUniform4dv");
    context->UniformMatrix2dv = (PFNGLUNIFORMMATRIX2DVPROC) load(userptr, "glUniformMatrix2dv");
    context->UniformMatrix2x3dv = (PFNGLUNIFORMMATRIX2X3DVPROC) load(userptr, "glUniformMatrix2x3dv");
    context->UniformMatrix2x4dv = (PFNGLUNIFORMMATRIX2X4DVPROC) load(userptr, "glUniformMatrix2x4dv");
    context->UniformMatrix3dv = (PFNGLUNIFORMMATRIX3DVPROC) load(userptr, "glUniformMatrix3dv");
    context->UniformMatrix3x2dv = (PFNGLUNIFORMMATRIX3X2DVPROC) load(userptr, "glUniformMatrix3x2dv");
    context->UniformMatrix3x4dv = (PFNGLUNIFORMMATRIX3X4DVPROC) load(userptr, "glUniformMatrix3x4dv");
    context->UniformMatrix4dv = (PFNGLUNIFORMMATRIX4DVPROC) load(userptr, "glUniformMatrix4dv");
    context->UniformMatrix4x2dv = (PFNGLUNIFORMMATRIX4X2DVPROC) load(userptr, "glUniformMatrix4x2dv");
    context->UniformMatrix4x3dv = (PFNGLUNIFORMMATRIX4X3DVPROC) load(userptr, "glUniformMatrix4x3dv");
}
static void glad_gl_load_GL_ARB_gpu_shader_int64(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_gpu_shader_int64) return;
    context->GetUniformi64vARB = (PFNGLGETUNIFORMI64VARBPROC) load(userptr, "glGetUniformi64vARB");
    context->GetUniformui64vARB = (PFNGLGETUNIFORMUI64VARBPROC) load(userptr, "glGetUniformui64vARB");
    context->GetnUniformi64vARB = (PFNGLGETNUNIFORMI64VARBPROC) load(userptr, "glGetnUniformi64vARB");
    context->GetnUniformui64vARB = (PFNGLGETNUNIFORMUI64VARBPROC) load(userptr, "glGetnUniformui64vARB");
    context->ProgramUniform1i64ARB = (PFNGLPROGRAMUNIFORM1I64ARBPROC) load(userptr, "glProgramUniform1i64ARB");
    context->ProgramUniform1i64vARB = (PFNGLPROGRAMUNIFORM1I64VARBPROC) load(userptr, "glProgramUniform1i64vARB");
    context->ProgramUniform1ui64ARB = (PFNGLPROGRAMUNIFORM1UI64ARBPROC) load(userptr, "glProgramUniform1ui64ARB");
    context->ProgramUniform1ui64vARB = (PFNGLPROGRAMUNIFORM1UI64VARBPROC) load(userptr, "glProgramUniform1ui64vARB");
    context->ProgramUniform2i64ARB = (PFNGLPROGRAMUNIFORM2I64ARBPROC) load(userptr, "glProgramUniform2i64ARB");
    context->ProgramUniform2i64vARB = (PFNGLPROGRAMUNIFORM2I64VARBPROC) load(userptr, "glProgramUniform2i64vARB");
    context->ProgramUniform2ui64ARB = (PFNGLPROGRAMUNIFORM2UI64ARBPROC) load(userptr, "glProgramUniform2ui64ARB");
    context->ProgramUniform2ui64vARB = (PFNGLPROGRAMUNIFORM2UI64VARBPROC) load(userptr, "glProgramUniform2ui64vARB");
    context->ProgramUniform3i64ARB = (PFNGLPROGRAMUNIFORM3I64ARBPROC) load(userptr, "glProgramUniform3i64ARB");
    context->ProgramUniform3i64vARB = (PFNGLPROGRAMUNIFORM3I64VARBPROC) load(userptr, "glProgramUniform3i64vARB");
    context->ProgramUniform3ui64ARB = (PFNGLPROGRAMUNIFORM3UI64ARBPROC) load(userptr, "glProgramUniform3ui64ARB");
    context->ProgramUniform3ui64vARB = (PFNGLPROGRAMUNIFORM3UI64VARBPROC) load(userptr, "glProgramUniform3ui64vARB");
    context->ProgramUniform4i64ARB = (PFNGLPROGRAMUNIFORM4I64ARBPROC) load(userptr, "glProgramUniform4i64ARB");
    context->ProgramUniform4i64vARB = (PFNGLPROGRAMUNIFORM4I64VARBPROC) load(userptr, "glProgramUniform4i64vARB");
    context->ProgramUniform4ui64ARB = (PFNGLPROGRAMUNIFORM4UI64ARBPROC) load(userptr, "glProgramUniform4ui64ARB");
    context->ProgramUniform4ui64vARB = (PFNGLPROGRAMUNIFORM4UI64VARBPROC) load(userptr, "glProgramUniform4ui64vARB");
    context->Uniform1i64ARB = (PFNGLUNIFORM1I64ARBPROC) load(userptr, "glUniform1i64ARB");
    context->Uniform1i64vARB = (PFNGLUNIFORM1I64VARBPROC) load(userptr, "glUniform1i64vARB");
    context->Uniform1ui64ARB = (PFNGLUNIFORM1UI64ARBPROC) load(userptr, "glUniform1ui64ARB");
    context->Uniform1ui64vARB = (PFNGLUNIFORM1UI64VARBPROC) load(userptr, "glUniform1ui64vARB");
    context->Uniform2i64ARB = (PFNGLUNIFORM2I64ARBPROC) load(userptr, "glUniform2i64ARB");
    context->Uniform2i64vARB = (PFNGLUNIFORM2I64VARBPROC) load(userptr, "glUniform2i64vARB");
    context->Uniform2ui64ARB = (PFNGLUNIFORM2UI64ARBPROC) load(userptr, "glUniform2ui64ARB");
    context->Uniform2ui64vARB = (PFNGLUNIFORM2UI64VARBPROC) load(userptr, "glUniform2ui64vARB");
    context->Uniform3i64ARB = (PFNGLUNIFORM3I64ARBPROC) load(userptr, "glUniform3i64ARB");
    context->Uniform3i64vARB = (PFNGLUNIFORM3I64VARBPROC) load(userptr, "glUniform3i64vARB");
    context->Uniform3ui64ARB = (PFNGLUNIFORM3UI64ARBPROC) load(userptr, "glUniform3ui64ARB");
    context->Uniform3ui64vARB = (PFNGLUNIFORM3UI64VARBPROC) load(userptr, "glUniform3ui64vARB");
    context->Uniform4i64ARB = (PFNGLUNIFORM4I64ARBPROC) load(userptr, "glUniform4i64ARB");
    context->Uniform4i64vARB = (PFNGLUNIFORM4I64VARBPROC) load(userptr, "glUniform4i64vARB");
    context->Uniform4ui64ARB = (PFNGLUNIFORM4UI64ARBPROC) load(userptr, "glUniform4ui64ARB");
    context->Uniform4ui64vARB = (PFNGLUNIFORM4UI64VARBPROC) load(userptr, "glUniform4ui64vARB");
}
static void glad_gl_load_GL_ARB_imaging(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_imaging) return;
    context->BlendColor = (PFNGLBLENDCOLORPROC) load(userptr, "glBlendColor");
    context->BlendEquation = (PFNGLBLENDEQUATIONPROC) load(userptr, "glBlendEquation");
    context->ColorSubTable = (PFNGLCOLORSUBTABLEPROC) load(userptr, "glColorSubTable");
    context->ColorTable = (PFNGLCOLORTABLEPROC) load(userptr, "glColorTable");
    context->ColorTableParameterfv = (PFNGLCOLORTABLEPARAMETERFVPROC) load(userptr, "glColorTableParameterfv");
    context->ColorTableParameteriv = (PFNGLCOLORTABLEPARAMETERIVPROC) load(userptr, "glColorTableParameteriv");
    context->ConvolutionFilter1D = (PFNGLCONVOLUTIONFILTER1DPROC) load(userptr, "glConvolutionFilter1D");
    context->ConvolutionFilter2D = (PFNGLCONVOLUTIONFILTER2DPROC) load(userptr, "glConvolutionFilter2D");
    context->ConvolutionParameterf = (PFNGLCONVOLUTIONPARAMETERFPROC) load(userptr, "glConvolutionParameterf");
    context->ConvolutionParameterfv = (PFNGLCONVOLUTIONPARAMETERFVPROC) load(userptr, "glConvolutionParameterfv");
    context->ConvolutionParameteri = (PFNGLCONVOLUTIONPARAMETERIPROC) load(userptr, "glConvolutionParameteri");
    context->ConvolutionParameteriv = (PFNGLCONVOLUTIONPARAMETERIVPROC) load(userptr, "glConvolutionParameteriv");
    context->CopyColorSubTable = (PFNGLCOPYCOLORSUBTABLEPROC) load(userptr, "glCopyColorSubTable");
    context->CopyColorTable = (PFNGLCOPYCOLORTABLEPROC) load(userptr, "glCopyColorTable");
    context->CopyConvolutionFilter1D = (PFNGLCOPYCONVOLUTIONFILTER1DPROC) load(userptr, "glCopyConvolutionFilter1D");
    context->CopyConvolutionFilter2D = (PFNGLCOPYCONVOLUTIONFILTER2DPROC) load(userptr, "glCopyConvolutionFilter2D");
    context->GetColorTable = (PFNGLGETCOLORTABLEPROC) load(userptr, "glGetColorTable");
    context->GetColorTableParameterfv = (PFNGLGETCOLORTABLEPARAMETERFVPROC) load(userptr, "glGetColorTableParameterfv");
    context->GetColorTableParameteriv = (PFNGLGETCOLORTABLEPARAMETERIVPROC) load(userptr, "glGetColorTableParameteriv");
    context->GetConvolutionFilter = (PFNGLGETCONVOLUTIONFILTERPROC) load(userptr, "glGetConvolutionFilter");
    context->GetConvolutionParameterfv = (PFNGLGETCONVOLUTIONPARAMETERFVPROC) load(userptr, "glGetConvolutionParameterfv");
    context->GetConvolutionParameteriv = (PFNGLGETCONVOLUTIONPARAMETERIVPROC) load(userptr, "glGetConvolutionParameteriv");
    context->GetHistogram = (PFNGLGETHISTOGRAMPROC) load(userptr, "glGetHistogram");
    context->GetHistogramParameterfv = (PFNGLGETHISTOGRAMPARAMETERFVPROC) load(userptr, "glGetHistogramParameterfv");
    context->GetHistogramParameteriv = (PFNGLGETHISTOGRAMPARAMETERIVPROC) load(userptr, "glGetHistogramParameteriv");
    context->GetMinmax = (PFNGLGETMINMAXPROC) load(userptr, "glGetMinmax");
    context->GetMinmaxParameterfv = (PFNGLGETMINMAXPARAMETERFVPROC) load(userptr, "glGetMinmaxParameterfv");
    context->GetMinmaxParameteriv = (PFNGLGETMINMAXPARAMETERIVPROC) load(userptr, "glGetMinmaxParameteriv");
    context->GetSeparableFilter = (PFNGLGETSEPARABLEFILTERPROC) load(userptr, "glGetSeparableFilter");
    context->Histogram = (PFNGLHISTOGRAMPROC) load(userptr, "glHistogram");
    context->Minmax = (PFNGLMINMAXPROC) load(userptr, "glMinmax");
    context->ResetHistogram = (PFNGLRESETHISTOGRAMPROC) load(userptr, "glResetHistogram");
    context->ResetMinmax = (PFNGLRESETMINMAXPROC) load(userptr, "glResetMinmax");
    context->SeparableFilter2D = (PFNGLSEPARABLEFILTER2DPROC) load(userptr, "glSeparableFilter2D");
}
static void glad_gl_load_GL_ARB_indirect_parameters(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_indirect_parameters) return;
    context->MultiDrawArraysIndirectCountARB = (PFNGLMULTIDRAWARRAYSINDIRECTCOUNTARBPROC) load(userptr, "glMultiDrawArraysIndirectCountARB");
    context->MultiDrawElementsIndirectCountARB = (PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTARBPROC) load(userptr, "glMultiDrawElementsIndirectCountARB");
}
static void glad_gl_load_GL_ARB_instanced_arrays(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_instanced_arrays) return;
    context->VertexAttribDivisorARB = (PFNGLVERTEXATTRIBDIVISORARBPROC) load(userptr, "glVertexAttribDivisorARB");
}
static void glad_gl_load_GL_ARB_internalformat_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_internalformat_query) return;
    context->GetInternalformativ = (PFNGLGETINTERNALFORMATIVPROC) load(userptr, "glGetInternalformativ");
}
static void glad_gl_load_GL_ARB_internalformat_query2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_internalformat_query2) return;
    context->GetInternalformati64v = (PFNGLGETINTERNALFORMATI64VPROC) load(userptr, "glGetInternalformati64v");
}
static void glad_gl_load_GL_ARB_invalidate_subdata(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_invalidate_subdata) return;
    context->InvalidateBufferData = (PFNGLINVALIDATEBUFFERDATAPROC) load(userptr, "glInvalidateBufferData");
    context->InvalidateBufferSubData = (PFNGLINVALIDATEBUFFERSUBDATAPROC) load(userptr, "glInvalidateBufferSubData");
    context->InvalidateFramebuffer = (PFNGLINVALIDATEFRAMEBUFFERPROC) load(userptr, "glInvalidateFramebuffer");
    context->InvalidateSubFramebuffer = (PFNGLINVALIDATESUBFRAMEBUFFERPROC) load(userptr, "glInvalidateSubFramebuffer");
    context->InvalidateTexImage = (PFNGLINVALIDATETEXIMAGEPROC) load(userptr, "glInvalidateTexImage");
    context->InvalidateTexSubImage = (PFNGLINVALIDATETEXSUBIMAGEPROC) load(userptr, "glInvalidateTexSubImage");
}
static void glad_gl_load_GL_ARB_map_buffer_range(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_map_buffer_range) return;
    context->FlushMappedBufferRange = (PFNGLFLUSHMAPPEDBUFFERRANGEPROC) load(userptr, "glFlushMappedBufferRange");
    context->MapBufferRange = (PFNGLMAPBUFFERRANGEPROC) load(userptr, "glMapBufferRange");
}
static void glad_gl_load_GL_ARB_matrix_palette(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_matrix_palette) return;
    context->CurrentPaletteMatrixARB = (PFNGLCURRENTPALETTEMATRIXARBPROC) load(userptr, "glCurrentPaletteMatrixARB");
    context->MatrixIndexPointerARB = (PFNGLMATRIXINDEXPOINTERARBPROC) load(userptr, "glMatrixIndexPointerARB");
    context->MatrixIndexubvARB = (PFNGLMATRIXINDEXUBVARBPROC) load(userptr, "glMatrixIndexubvARB");
    context->MatrixIndexuivARB = (PFNGLMATRIXINDEXUIVARBPROC) load(userptr, "glMatrixIndexuivARB");
    context->MatrixIndexusvARB = (PFNGLMATRIXINDEXUSVARBPROC) load(userptr, "glMatrixIndexusvARB");
}
static void glad_gl_load_GL_ARB_multi_bind(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_multi_bind) return;
    context->BindBuffersBase = (PFNGLBINDBUFFERSBASEPROC) load(userptr, "glBindBuffersBase");
    context->BindBuffersRange = (PFNGLBINDBUFFERSRANGEPROC) load(userptr, "glBindBuffersRange");
    context->BindImageTextures = (PFNGLBINDIMAGETEXTURESPROC) load(userptr, "glBindImageTextures");
    context->BindSamplers = (PFNGLBINDSAMPLERSPROC) load(userptr, "glBindSamplers");
    context->BindTextures = (PFNGLBINDTEXTURESPROC) load(userptr, "glBindTextures");
    context->BindVertexBuffers = (PFNGLBINDVERTEXBUFFERSPROC) load(userptr, "glBindVertexBuffers");
}
static void glad_gl_load_GL_ARB_multi_draw_indirect(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_multi_draw_indirect) return;
    context->MultiDrawArraysIndirect = (PFNGLMULTIDRAWARRAYSINDIRECTPROC) load(userptr, "glMultiDrawArraysIndirect");
    context->MultiDrawElementsIndirect = (PFNGLMULTIDRAWELEMENTSINDIRECTPROC) load(userptr, "glMultiDrawElementsIndirect");
}
static void glad_gl_load_GL_ARB_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_multisample) return;
    context->SampleCoverageARB = (PFNGLSAMPLECOVERAGEARBPROC) load(userptr, "glSampleCoverageARB");
}
static void glad_gl_load_GL_ARB_multitexture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_multitexture) return;
    context->ActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC) load(userptr, "glActiveTextureARB");
    context->ClientActiveTextureARB = (PFNGLCLIENTACTIVETEXTUREARBPROC) load(userptr, "glClientActiveTextureARB");
    context->MultiTexCoord1dARB = (PFNGLMULTITEXCOORD1DARBPROC) load(userptr, "glMultiTexCoord1dARB");
    context->MultiTexCoord1dvARB = (PFNGLMULTITEXCOORD1DVARBPROC) load(userptr, "glMultiTexCoord1dvARB");
    context->MultiTexCoord1fARB = (PFNGLMULTITEXCOORD1FARBPROC) load(userptr, "glMultiTexCoord1fARB");
    context->MultiTexCoord1fvARB = (PFNGLMULTITEXCOORD1FVARBPROC) load(userptr, "glMultiTexCoord1fvARB");
    context->MultiTexCoord1iARB = (PFNGLMULTITEXCOORD1IARBPROC) load(userptr, "glMultiTexCoord1iARB");
    context->MultiTexCoord1ivARB = (PFNGLMULTITEXCOORD1IVARBPROC) load(userptr, "glMultiTexCoord1ivARB");
    context->MultiTexCoord1sARB = (PFNGLMULTITEXCOORD1SARBPROC) load(userptr, "glMultiTexCoord1sARB");
    context->MultiTexCoord1svARB = (PFNGLMULTITEXCOORD1SVARBPROC) load(userptr, "glMultiTexCoord1svARB");
    context->MultiTexCoord2dARB = (PFNGLMULTITEXCOORD2DARBPROC) load(userptr, "glMultiTexCoord2dARB");
    context->MultiTexCoord2dvARB = (PFNGLMULTITEXCOORD2DVARBPROC) load(userptr, "glMultiTexCoord2dvARB");
    context->MultiTexCoord2fARB = (PFNGLMULTITEXCOORD2FARBPROC) load(userptr, "glMultiTexCoord2fARB");
    context->MultiTexCoord2fvARB = (PFNGLMULTITEXCOORD2FVARBPROC) load(userptr, "glMultiTexCoord2fvARB");
    context->MultiTexCoord2iARB = (PFNGLMULTITEXCOORD2IARBPROC) load(userptr, "glMultiTexCoord2iARB");
    context->MultiTexCoord2ivARB = (PFNGLMULTITEXCOORD2IVARBPROC) load(userptr, "glMultiTexCoord2ivARB");
    context->MultiTexCoord2sARB = (PFNGLMULTITEXCOORD2SARBPROC) load(userptr, "glMultiTexCoord2sARB");
    context->MultiTexCoord2svARB = (PFNGLMULTITEXCOORD2SVARBPROC) load(userptr, "glMultiTexCoord2svARB");
    context->MultiTexCoord3dARB = (PFNGLMULTITEXCOORD3DARBPROC) load(userptr, "glMultiTexCoord3dARB");
    context->MultiTexCoord3dvARB = (PFNGLMULTITEXCOORD3DVARBPROC) load(userptr, "glMultiTexCoord3dvARB");
    context->MultiTexCoord3fARB = (PFNGLMULTITEXCOORD3FARBPROC) load(userptr, "glMultiTexCoord3fARB");
    context->MultiTexCoord3fvARB = (PFNGLMULTITEXCOORD3FVARBPROC) load(userptr, "glMultiTexCoord3fvARB");
    context->MultiTexCoord3iARB = (PFNGLMULTITEXCOORD3IARBPROC) load(userptr, "glMultiTexCoord3iARB");
    context->MultiTexCoord3ivARB = (PFNGLMULTITEXCOORD3IVARBPROC) load(userptr, "glMultiTexCoord3ivARB");
    context->MultiTexCoord3sARB = (PFNGLMULTITEXCOORD3SARBPROC) load(userptr, "glMultiTexCoord3sARB");
    context->MultiTexCoord3svARB = (PFNGLMULTITEXCOORD3SVARBPROC) load(userptr, "glMultiTexCoord3svARB");
    context->MultiTexCoord4dARB = (PFNGLMULTITEXCOORD4DARBPROC) load(userptr, "glMultiTexCoord4dARB");
    context->MultiTexCoord4dvARB = (PFNGLMULTITEXCOORD4DVARBPROC) load(userptr, "glMultiTexCoord4dvARB");
    context->MultiTexCoord4fARB = (PFNGLMULTITEXCOORD4FARBPROC) load(userptr, "glMultiTexCoord4fARB");
    context->MultiTexCoord4fvARB = (PFNGLMULTITEXCOORD4FVARBPROC) load(userptr, "glMultiTexCoord4fvARB");
    context->MultiTexCoord4iARB = (PFNGLMULTITEXCOORD4IARBPROC) load(userptr, "glMultiTexCoord4iARB");
    context->MultiTexCoord4ivARB = (PFNGLMULTITEXCOORD4IVARBPROC) load(userptr, "glMultiTexCoord4ivARB");
    context->MultiTexCoord4sARB = (PFNGLMULTITEXCOORD4SARBPROC) load(userptr, "glMultiTexCoord4sARB");
    context->MultiTexCoord4svARB = (PFNGLMULTITEXCOORD4SVARBPROC) load(userptr, "glMultiTexCoord4svARB");
}
static void glad_gl_load_GL_ARB_occlusion_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_occlusion_query) return;
    context->BeginQueryARB = (PFNGLBEGINQUERYARBPROC) load(userptr, "glBeginQueryARB");
    context->DeleteQueriesARB = (PFNGLDELETEQUERIESARBPROC) load(userptr, "glDeleteQueriesARB");
    context->EndQueryARB = (PFNGLENDQUERYARBPROC) load(userptr, "glEndQueryARB");
    context->GenQueriesARB = (PFNGLGENQUERIESARBPROC) load(userptr, "glGenQueriesARB");
    context->GetQueryObjectivARB = (PFNGLGETQUERYOBJECTIVARBPROC) load(userptr, "glGetQueryObjectivARB");
    context->GetQueryObjectuivARB = (PFNGLGETQUERYOBJECTUIVARBPROC) load(userptr, "glGetQueryObjectuivARB");
    context->GetQueryivARB = (PFNGLGETQUERYIVARBPROC) load(userptr, "glGetQueryivARB");
    context->IsQueryARB = (PFNGLISQUERYARBPROC) load(userptr, "glIsQueryARB");
}
static void glad_gl_load_GL_ARB_parallel_shader_compile(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_parallel_shader_compile) return;
    context->MaxShaderCompilerThreadsARB = (PFNGLMAXSHADERCOMPILERTHREADSARBPROC) load(userptr, "glMaxShaderCompilerThreadsARB");
}
static void glad_gl_load_GL_ARB_point_parameters(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_point_parameters) return;
    context->PointParameterfARB = (PFNGLPOINTPARAMETERFARBPROC) load(userptr, "glPointParameterfARB");
    context->PointParameterfvARB = (PFNGLPOINTPARAMETERFVARBPROC) load(userptr, "glPointParameterfvARB");
}
static void glad_gl_load_GL_ARB_polygon_offset_clamp(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_polygon_offset_clamp) return;
    context->PolygonOffsetClamp = (PFNGLPOLYGONOFFSETCLAMPPROC) load(userptr, "glPolygonOffsetClamp");
}
static void glad_gl_load_GL_ARB_program_interface_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_program_interface_query) return;
    context->GetProgramInterfaceiv = (PFNGLGETPROGRAMINTERFACEIVPROC) load(userptr, "glGetProgramInterfaceiv");
    context->GetProgramResourceIndex = (PFNGLGETPROGRAMRESOURCEINDEXPROC) load(userptr, "glGetProgramResourceIndex");
    context->GetProgramResourceLocation = (PFNGLGETPROGRAMRESOURCELOCATIONPROC) load(userptr, "glGetProgramResourceLocation");
    context->GetProgramResourceLocationIndex = (PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC) load(userptr, "glGetProgramResourceLocationIndex");
    context->GetProgramResourceName = (PFNGLGETPROGRAMRESOURCENAMEPROC) load(userptr, "glGetProgramResourceName");
    context->GetProgramResourceiv = (PFNGLGETPROGRAMRESOURCEIVPROC) load(userptr, "glGetProgramResourceiv");
}
static void glad_gl_load_GL_ARB_provoking_vertex(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_provoking_vertex) return;
    context->ProvokingVertex = (PFNGLPROVOKINGVERTEXPROC) load(userptr, "glProvokingVertex");
}
static void glad_gl_load_GL_ARB_robustness(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_robustness) return;
    context->GetGraphicsResetStatusARB = (PFNGLGETGRAPHICSRESETSTATUSARBPROC) load(userptr, "glGetGraphicsResetStatusARB");
    context->GetnColorTableARB = (PFNGLGETNCOLORTABLEARBPROC) load(userptr, "glGetnColorTableARB");
    context->GetnCompressedTexImageARB = (PFNGLGETNCOMPRESSEDTEXIMAGEARBPROC) load(userptr, "glGetnCompressedTexImageARB");
    context->GetnConvolutionFilterARB = (PFNGLGETNCONVOLUTIONFILTERARBPROC) load(userptr, "glGetnConvolutionFilterARB");
    context->GetnHistogramARB = (PFNGLGETNHISTOGRAMARBPROC) load(userptr, "glGetnHistogramARB");
    context->GetnMapdvARB = (PFNGLGETNMAPDVARBPROC) load(userptr, "glGetnMapdvARB");
    context->GetnMapfvARB = (PFNGLGETNMAPFVARBPROC) load(userptr, "glGetnMapfvARB");
    context->GetnMapivARB = (PFNGLGETNMAPIVARBPROC) load(userptr, "glGetnMapivARB");
    context->GetnMinmaxARB = (PFNGLGETNMINMAXARBPROC) load(userptr, "glGetnMinmaxARB");
    context->GetnPixelMapfvARB = (PFNGLGETNPIXELMAPFVARBPROC) load(userptr, "glGetnPixelMapfvARB");
    context->GetnPixelMapuivARB = (PFNGLGETNPIXELMAPUIVARBPROC) load(userptr, "glGetnPixelMapuivARB");
    context->GetnPixelMapusvARB = (PFNGLGETNPIXELMAPUSVARBPROC) load(userptr, "glGetnPixelMapusvARB");
    context->GetnPolygonStippleARB = (PFNGLGETNPOLYGONSTIPPLEARBPROC) load(userptr, "glGetnPolygonStippleARB");
    context->GetnSeparableFilterARB = (PFNGLGETNSEPARABLEFILTERARBPROC) load(userptr, "glGetnSeparableFilterARB");
    context->GetnTexImageARB = (PFNGLGETNTEXIMAGEARBPROC) load(userptr, "glGetnTexImageARB");
    context->GetnUniformdvARB = (PFNGLGETNUNIFORMDVARBPROC) load(userptr, "glGetnUniformdvARB");
    context->GetnUniformfvARB = (PFNGLGETNUNIFORMFVARBPROC) load(userptr, "glGetnUniformfvARB");
    context->GetnUniformivARB = (PFNGLGETNUNIFORMIVARBPROC) load(userptr, "glGetnUniformivARB");
    context->GetnUniformuivARB = (PFNGLGETNUNIFORMUIVARBPROC) load(userptr, "glGetnUniformuivARB");
    context->ReadnPixelsARB = (PFNGLREADNPIXELSARBPROC) load(userptr, "glReadnPixelsARB");
}
static void glad_gl_load_GL_ARB_sample_locations(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_sample_locations) return;
    context->EvaluateDepthValuesARB = (PFNGLEVALUATEDEPTHVALUESARBPROC) load(userptr, "glEvaluateDepthValuesARB");
    context->FramebufferSampleLocationsfvARB = (PFNGLFRAMEBUFFERSAMPLELOCATIONSFVARBPROC) load(userptr, "glFramebufferSampleLocationsfvARB");
    context->NamedFramebufferSampleLocationsfvARB = (PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVARBPROC) load(userptr, "glNamedFramebufferSampleLocationsfvARB");
}
static void glad_gl_load_GL_ARB_sample_shading(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_sample_shading) return;
    context->MinSampleShadingARB = (PFNGLMINSAMPLESHADINGARBPROC) load(userptr, "glMinSampleShadingARB");
}
static void glad_gl_load_GL_ARB_sampler_objects(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_sampler_objects) return;
    context->BindSampler = (PFNGLBINDSAMPLERPROC) load(userptr, "glBindSampler");
    context->DeleteSamplers = (PFNGLDELETESAMPLERSPROC) load(userptr, "glDeleteSamplers");
    context->GenSamplers = (PFNGLGENSAMPLERSPROC) load(userptr, "glGenSamplers");
    context->GetSamplerParameterIiv = (PFNGLGETSAMPLERPARAMETERIIVPROC) load(userptr, "glGetSamplerParameterIiv");
    context->GetSamplerParameterIuiv = (PFNGLGETSAMPLERPARAMETERIUIVPROC) load(userptr, "glGetSamplerParameterIuiv");
    context->GetSamplerParameterfv = (PFNGLGETSAMPLERPARAMETERFVPROC) load(userptr, "glGetSamplerParameterfv");
    context->GetSamplerParameteriv = (PFNGLGETSAMPLERPARAMETERIVPROC) load(userptr, "glGetSamplerParameteriv");
    context->IsSampler = (PFNGLISSAMPLERPROC) load(userptr, "glIsSampler");
    context->SamplerParameterIiv = (PFNGLSAMPLERPARAMETERIIVPROC) load(userptr, "glSamplerParameterIiv");
    context->SamplerParameterIuiv = (PFNGLSAMPLERPARAMETERIUIVPROC) load(userptr, "glSamplerParameterIuiv");
    context->SamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC) load(userptr, "glSamplerParameterf");
    context->SamplerParameterfv = (PFNGLSAMPLERPARAMETERFVPROC) load(userptr, "glSamplerParameterfv");
    context->SamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC) load(userptr, "glSamplerParameteri");
    context->SamplerParameteriv = (PFNGLSAMPLERPARAMETERIVPROC) load(userptr, "glSamplerParameteriv");
}
static void glad_gl_load_GL_ARB_separate_shader_objects(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_separate_shader_objects) return;
    context->ActiveShaderProgram = (PFNGLACTIVESHADERPROGRAMPROC) load(userptr, "glActiveShaderProgram");
    context->BindProgramPipeline = (PFNGLBINDPROGRAMPIPELINEPROC) load(userptr, "glBindProgramPipeline");
    context->CreateShaderProgramv = (PFNGLCREATESHADERPROGRAMVPROC) load(userptr, "glCreateShaderProgramv");
    context->DeleteProgramPipelines = (PFNGLDELETEPROGRAMPIPELINESPROC) load(userptr, "glDeleteProgramPipelines");
    context->GenProgramPipelines = (PFNGLGENPROGRAMPIPELINESPROC) load(userptr, "glGenProgramPipelines");
    context->GetProgramPipelineInfoLog = (PFNGLGETPROGRAMPIPELINEINFOLOGPROC) load(userptr, "glGetProgramPipelineInfoLog");
    context->GetProgramPipelineiv = (PFNGLGETPROGRAMPIPELINEIVPROC) load(userptr, "glGetProgramPipelineiv");
    context->IsProgramPipeline = (PFNGLISPROGRAMPIPELINEPROC) load(userptr, "glIsProgramPipeline");
    context->ProgramParameteri = (PFNGLPROGRAMPARAMETERIPROC) load(userptr, "glProgramParameteri");
    context->ProgramUniform1d = (PFNGLPROGRAMUNIFORM1DPROC) load(userptr, "glProgramUniform1d");
    context->ProgramUniform1dv = (PFNGLPROGRAMUNIFORM1DVPROC) load(userptr, "glProgramUniform1dv");
    context->ProgramUniform1f = (PFNGLPROGRAMUNIFORM1FPROC) load(userptr, "glProgramUniform1f");
    context->ProgramUniform1fv = (PFNGLPROGRAMUNIFORM1FVPROC) load(userptr, "glProgramUniform1fv");
    context->ProgramUniform1i = (PFNGLPROGRAMUNIFORM1IPROC) load(userptr, "glProgramUniform1i");
    context->ProgramUniform1iv = (PFNGLPROGRAMUNIFORM1IVPROC) load(userptr, "glProgramUniform1iv");
    context->ProgramUniform1ui = (PFNGLPROGRAMUNIFORM1UIPROC) load(userptr, "glProgramUniform1ui");
    context->ProgramUniform1uiv = (PFNGLPROGRAMUNIFORM1UIVPROC) load(userptr, "glProgramUniform1uiv");
    context->ProgramUniform2d = (PFNGLPROGRAMUNIFORM2DPROC) load(userptr, "glProgramUniform2d");
    context->ProgramUniform2dv = (PFNGLPROGRAMUNIFORM2DVPROC) load(userptr, "glProgramUniform2dv");
    context->ProgramUniform2f = (PFNGLPROGRAMUNIFORM2FPROC) load(userptr, "glProgramUniform2f");
    context->ProgramUniform2fv = (PFNGLPROGRAMUNIFORM2FVPROC) load(userptr, "glProgramUniform2fv");
    context->ProgramUniform2i = (PFNGLPROGRAMUNIFORM2IPROC) load(userptr, "glProgramUniform2i");
    context->ProgramUniform2iv = (PFNGLPROGRAMUNIFORM2IVPROC) load(userptr, "glProgramUniform2iv");
    context->ProgramUniform2ui = (PFNGLPROGRAMUNIFORM2UIPROC) load(userptr, "glProgramUniform2ui");
    context->ProgramUniform2uiv = (PFNGLPROGRAMUNIFORM2UIVPROC) load(userptr, "glProgramUniform2uiv");
    context->ProgramUniform3d = (PFNGLPROGRAMUNIFORM3DPROC) load(userptr, "glProgramUniform3d");
    context->ProgramUniform3dv = (PFNGLPROGRAMUNIFORM3DVPROC) load(userptr, "glProgramUniform3dv");
    context->ProgramUniform3f = (PFNGLPROGRAMUNIFORM3FPROC) load(userptr, "glProgramUniform3f");
    context->ProgramUniform3fv = (PFNGLPROGRAMUNIFORM3FVPROC) load(userptr, "glProgramUniform3fv");
    context->ProgramUniform3i = (PFNGLPROGRAMUNIFORM3IPROC) load(userptr, "glProgramUniform3i");
    context->ProgramUniform3iv = (PFNGLPROGRAMUNIFORM3IVPROC) load(userptr, "glProgramUniform3iv");
    context->ProgramUniform3ui = (PFNGLPROGRAMUNIFORM3UIPROC) load(userptr, "glProgramUniform3ui");
    context->ProgramUniform3uiv = (PFNGLPROGRAMUNIFORM3UIVPROC) load(userptr, "glProgramUniform3uiv");
    context->ProgramUniform4d = (PFNGLPROGRAMUNIFORM4DPROC) load(userptr, "glProgramUniform4d");
    context->ProgramUniform4dv = (PFNGLPROGRAMUNIFORM4DVPROC) load(userptr, "glProgramUniform4dv");
    context->ProgramUniform4f = (PFNGLPROGRAMUNIFORM4FPROC) load(userptr, "glProgramUniform4f");
    context->ProgramUniform4fv = (PFNGLPROGRAMUNIFORM4FVPROC) load(userptr, "glProgramUniform4fv");
    context->ProgramUniform4i = (PFNGLPROGRAMUNIFORM4IPROC) load(userptr, "glProgramUniform4i");
    context->ProgramUniform4iv = (PFNGLPROGRAMUNIFORM4IVPROC) load(userptr, "glProgramUniform4iv");
    context->ProgramUniform4ui = (PFNGLPROGRAMUNIFORM4UIPROC) load(userptr, "glProgramUniform4ui");
    context->ProgramUniform4uiv = (PFNGLPROGRAMUNIFORM4UIVPROC) load(userptr, "glProgramUniform4uiv");
    context->ProgramUniformMatrix2dv = (PFNGLPROGRAMUNIFORMMATRIX2DVPROC) load(userptr, "glProgramUniformMatrix2dv");
    context->ProgramUniformMatrix2fv = (PFNGLPROGRAMUNIFORMMATRIX2FVPROC) load(userptr, "glProgramUniformMatrix2fv");
    context->ProgramUniformMatrix2x3dv = (PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC) load(userptr, "glProgramUniformMatrix2x3dv");
    context->ProgramUniformMatrix2x3fv = (PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC) load(userptr, "glProgramUniformMatrix2x3fv");
    context->ProgramUniformMatrix2x4dv = (PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC) load(userptr, "glProgramUniformMatrix2x4dv");
    context->ProgramUniformMatrix2x4fv = (PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC) load(userptr, "glProgramUniformMatrix2x4fv");
    context->ProgramUniformMatrix3dv = (PFNGLPROGRAMUNIFORMMATRIX3DVPROC) load(userptr, "glProgramUniformMatrix3dv");
    context->ProgramUniformMatrix3fv = (PFNGLPROGRAMUNIFORMMATRIX3FVPROC) load(userptr, "glProgramUniformMatrix3fv");
    context->ProgramUniformMatrix3x2dv = (PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC) load(userptr, "glProgramUniformMatrix3x2dv");
    context->ProgramUniformMatrix3x2fv = (PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC) load(userptr, "glProgramUniformMatrix3x2fv");
    context->ProgramUniformMatrix3x4dv = (PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC) load(userptr, "glProgramUniformMatrix3x4dv");
    context->ProgramUniformMatrix3x4fv = (PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC) load(userptr, "glProgramUniformMatrix3x4fv");
    context->ProgramUniformMatrix4dv = (PFNGLPROGRAMUNIFORMMATRIX4DVPROC) load(userptr, "glProgramUniformMatrix4dv");
    context->ProgramUniformMatrix4fv = (PFNGLPROGRAMUNIFORMMATRIX4FVPROC) load(userptr, "glProgramUniformMatrix4fv");
    context->ProgramUniformMatrix4x2dv = (PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC) load(userptr, "glProgramUniformMatrix4x2dv");
    context->ProgramUniformMatrix4x2fv = (PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC) load(userptr, "glProgramUniformMatrix4x2fv");
    context->ProgramUniformMatrix4x3dv = (PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC) load(userptr, "glProgramUniformMatrix4x3dv");
    context->ProgramUniformMatrix4x3fv = (PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC) load(userptr, "glProgramUniformMatrix4x3fv");
    context->UseProgramStages = (PFNGLUSEPROGRAMSTAGESPROC) load(userptr, "glUseProgramStages");
    context->ValidateProgramPipeline = (PFNGLVALIDATEPROGRAMPIPELINEPROC) load(userptr, "glValidateProgramPipeline");
}
static void glad_gl_load_GL_ARB_shader_atomic_counters(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_shader_atomic_counters) return;
    context->GetActiveAtomicCounterBufferiv = (PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC) load(userptr, "glGetActiveAtomicCounterBufferiv");
}
static void glad_gl_load_GL_ARB_shader_image_load_store(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_shader_image_load_store) return;
    context->BindImageTexture = (PFNGLBINDIMAGETEXTUREPROC) load(userptr, "glBindImageTexture");
    context->MemoryBarrier = (PFNGLMEMORYBARRIERPROC) load(userptr, "glMemoryBarrier");
}
static void glad_gl_load_GL_ARB_shader_objects(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_shader_objects) return;
    context->AttachObjectARB = (PFNGLATTACHOBJECTARBPROC) load(userptr, "glAttachObjectARB");
    context->CompileShaderARB = (PFNGLCOMPILESHADERARBPROC) load(userptr, "glCompileShaderARB");
    context->CreateProgramObjectARB = (PFNGLCREATEPROGRAMOBJECTARBPROC) load(userptr, "glCreateProgramObjectARB");
    context->CreateShaderObjectARB = (PFNGLCREATESHADEROBJECTARBPROC) load(userptr, "glCreateShaderObjectARB");
    context->DeleteObjectARB = (PFNGLDELETEOBJECTARBPROC) load(userptr, "glDeleteObjectARB");
    context->DetachObjectARB = (PFNGLDETACHOBJECTARBPROC) load(userptr, "glDetachObjectARB");
    context->GetActiveUniformARB = (PFNGLGETACTIVEUNIFORMARBPROC) load(userptr, "glGetActiveUniformARB");
    context->GetAttachedObjectsARB = (PFNGLGETATTACHEDOBJECTSARBPROC) load(userptr, "glGetAttachedObjectsARB");
    context->GetHandleARB = (PFNGLGETHANDLEARBPROC) load(userptr, "glGetHandleARB");
    context->GetInfoLogARB = (PFNGLGETINFOLOGARBPROC) load(userptr, "glGetInfoLogARB");
    context->GetObjectParameterfvARB = (PFNGLGETOBJECTPARAMETERFVARBPROC) load(userptr, "glGetObjectParameterfvARB");
    context->GetObjectParameterivARB = (PFNGLGETOBJECTPARAMETERIVARBPROC) load(userptr, "glGetObjectParameterivARB");
    context->GetShaderSourceARB = (PFNGLGETSHADERSOURCEARBPROC) load(userptr, "glGetShaderSourceARB");
    context->GetUniformLocationARB = (PFNGLGETUNIFORMLOCATIONARBPROC) load(userptr, "glGetUniformLocationARB");
    context->GetUniformfvARB = (PFNGLGETUNIFORMFVARBPROC) load(userptr, "glGetUniformfvARB");
    context->GetUniformivARB = (PFNGLGETUNIFORMIVARBPROC) load(userptr, "glGetUniformivARB");
    context->LinkProgramARB = (PFNGLLINKPROGRAMARBPROC) load(userptr, "glLinkProgramARB");
    context->ShaderSourceARB = (PFNGLSHADERSOURCEARBPROC) load(userptr, "glShaderSourceARB");
    context->Uniform1fARB = (PFNGLUNIFORM1FARBPROC) load(userptr, "glUniform1fARB");
    context->Uniform1fvARB = (PFNGLUNIFORM1FVARBPROC) load(userptr, "glUniform1fvARB");
    context->Uniform1iARB = (PFNGLUNIFORM1IARBPROC) load(userptr, "glUniform1iARB");
    context->Uniform1ivARB = (PFNGLUNIFORM1IVARBPROC) load(userptr, "glUniform1ivARB");
    context->Uniform2fARB = (PFNGLUNIFORM2FARBPROC) load(userptr, "glUniform2fARB");
    context->Uniform2fvARB = (PFNGLUNIFORM2FVARBPROC) load(userptr, "glUniform2fvARB");
    context->Uniform2iARB = (PFNGLUNIFORM2IARBPROC) load(userptr, "glUniform2iARB");
    context->Uniform2ivARB = (PFNGLUNIFORM2IVARBPROC) load(userptr, "glUniform2ivARB");
    context->Uniform3fARB = (PFNGLUNIFORM3FARBPROC) load(userptr, "glUniform3fARB");
    context->Uniform3fvARB = (PFNGLUNIFORM3FVARBPROC) load(userptr, "glUniform3fvARB");
    context->Uniform3iARB = (PFNGLUNIFORM3IARBPROC) load(userptr, "glUniform3iARB");
    context->Uniform3ivARB = (PFNGLUNIFORM3IVARBPROC) load(userptr, "glUniform3ivARB");
    context->Uniform4fARB = (PFNGLUNIFORM4FARBPROC) load(userptr, "glUniform4fARB");
    context->Uniform4fvARB = (PFNGLUNIFORM4FVARBPROC) load(userptr, "glUniform4fvARB");
    context->Uniform4iARB = (PFNGLUNIFORM4IARBPROC) load(userptr, "glUniform4iARB");
    context->Uniform4ivARB = (PFNGLUNIFORM4IVARBPROC) load(userptr, "glUniform4ivARB");
    context->UniformMatrix2fvARB = (PFNGLUNIFORMMATRIX2FVARBPROC) load(userptr, "glUniformMatrix2fvARB");
    context->UniformMatrix3fvARB = (PFNGLUNIFORMMATRIX3FVARBPROC) load(userptr, "glUniformMatrix3fvARB");
    context->UniformMatrix4fvARB = (PFNGLUNIFORMMATRIX4FVARBPROC) load(userptr, "glUniformMatrix4fvARB");
    context->UseProgramObjectARB = (PFNGLUSEPROGRAMOBJECTARBPROC) load(userptr, "glUseProgramObjectARB");
    context->ValidateProgramARB = (PFNGLVALIDATEPROGRAMARBPROC) load(userptr, "glValidateProgramARB");
}
static void glad_gl_load_GL_ARB_shader_storage_buffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_shader_storage_buffer_object) return;
    context->ShaderStorageBlockBinding = (PFNGLSHADERSTORAGEBLOCKBINDINGPROC) load(userptr, "glShaderStorageBlockBinding");
}
static void glad_gl_load_GL_ARB_shader_subroutine(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_shader_subroutine) return;
    context->GetActiveSubroutineName = (PFNGLGETACTIVESUBROUTINENAMEPROC) load(userptr, "glGetActiveSubroutineName");
    context->GetActiveSubroutineUniformName = (PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC) load(userptr, "glGetActiveSubroutineUniformName");
    context->GetActiveSubroutineUniformiv = (PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC) load(userptr, "glGetActiveSubroutineUniformiv");
    context->GetProgramStageiv = (PFNGLGETPROGRAMSTAGEIVPROC) load(userptr, "glGetProgramStageiv");
    context->GetSubroutineIndex = (PFNGLGETSUBROUTINEINDEXPROC) load(userptr, "glGetSubroutineIndex");
    context->GetSubroutineUniformLocation = (PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC) load(userptr, "glGetSubroutineUniformLocation");
    context->GetUniformSubroutineuiv = (PFNGLGETUNIFORMSUBROUTINEUIVPROC) load(userptr, "glGetUniformSubroutineuiv");
    context->UniformSubroutinesuiv = (PFNGLUNIFORMSUBROUTINESUIVPROC) load(userptr, "glUniformSubroutinesuiv");
}
static void glad_gl_load_GL_ARB_shading_language_include(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_shading_language_include) return;
    context->CompileShaderIncludeARB = (PFNGLCOMPILESHADERINCLUDEARBPROC) load(userptr, "glCompileShaderIncludeARB");
    context->DeleteNamedStringARB = (PFNGLDELETENAMEDSTRINGARBPROC) load(userptr, "glDeleteNamedStringARB");
    context->GetNamedStringARB = (PFNGLGETNAMEDSTRINGARBPROC) load(userptr, "glGetNamedStringARB");
    context->GetNamedStringivARB = (PFNGLGETNAMEDSTRINGIVARBPROC) load(userptr, "glGetNamedStringivARB");
    context->IsNamedStringARB = (PFNGLISNAMEDSTRINGARBPROC) load(userptr, "glIsNamedStringARB");
    context->NamedStringARB = (PFNGLNAMEDSTRINGARBPROC) load(userptr, "glNamedStringARB");
}
static void glad_gl_load_GL_ARB_sparse_buffer(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_sparse_buffer) return;
    context->BufferPageCommitmentARB = (PFNGLBUFFERPAGECOMMITMENTARBPROC) load(userptr, "glBufferPageCommitmentARB");
    context->NamedBufferPageCommitmentARB = (PFNGLNAMEDBUFFERPAGECOMMITMENTARBPROC) load(userptr, "glNamedBufferPageCommitmentARB");
    context->NamedBufferPageCommitmentEXT = (PFNGLNAMEDBUFFERPAGECOMMITMENTEXTPROC) load(userptr, "glNamedBufferPageCommitmentEXT");
}
static void glad_gl_load_GL_ARB_sparse_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_sparse_texture) return;
    context->TexPageCommitmentARB = (PFNGLTEXPAGECOMMITMENTARBPROC) load(userptr, "glTexPageCommitmentARB");
}
static void glad_gl_load_GL_ARB_sync(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_sync) return;
    context->ClientWaitSync = (PFNGLCLIENTWAITSYNCPROC) load(userptr, "glClientWaitSync");
    context->DeleteSync = (PFNGLDELETESYNCPROC) load(userptr, "glDeleteSync");
    context->FenceSync = (PFNGLFENCESYNCPROC) load(userptr, "glFenceSync");
    context->GetInteger64v = (PFNGLGETINTEGER64VPROC) load(userptr, "glGetInteger64v");
    context->GetSynciv = (PFNGLGETSYNCIVPROC) load(userptr, "glGetSynciv");
    context->IsSync = (PFNGLISSYNCPROC) load(userptr, "glIsSync");
    context->WaitSync = (PFNGLWAITSYNCPROC) load(userptr, "glWaitSync");
}
static void glad_gl_load_GL_ARB_tessellation_shader(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_tessellation_shader) return;
    context->PatchParameterfv = (PFNGLPATCHPARAMETERFVPROC) load(userptr, "glPatchParameterfv");
    context->PatchParameteri = (PFNGLPATCHPARAMETERIPROC) load(userptr, "glPatchParameteri");
}
static void glad_gl_load_GL_ARB_texture_barrier(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_barrier) return;
    context->TextureBarrier = (PFNGLTEXTUREBARRIERPROC) load(userptr, "glTextureBarrier");
}
static void glad_gl_load_GL_ARB_texture_buffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_buffer_object) return;
    context->TexBufferARB = (PFNGLTEXBUFFERARBPROC) load(userptr, "glTexBufferARB");
}
static void glad_gl_load_GL_ARB_texture_buffer_range(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_buffer_range) return;
    context->TexBufferRange = (PFNGLTEXBUFFERRANGEPROC) load(userptr, "glTexBufferRange");
}
static void glad_gl_load_GL_ARB_texture_compression(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_compression) return;
    context->CompressedTexImage1DARB = (PFNGLCOMPRESSEDTEXIMAGE1DARBPROC) load(userptr, "glCompressedTexImage1DARB");
    context->CompressedTexImage2DARB = (PFNGLCOMPRESSEDTEXIMAGE2DARBPROC) load(userptr, "glCompressedTexImage2DARB");
    context->CompressedTexImage3DARB = (PFNGLCOMPRESSEDTEXIMAGE3DARBPROC) load(userptr, "glCompressedTexImage3DARB");
    context->CompressedTexSubImage1DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC) load(userptr, "glCompressedTexSubImage1DARB");
    context->CompressedTexSubImage2DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC) load(userptr, "glCompressedTexSubImage2DARB");
    context->CompressedTexSubImage3DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC) load(userptr, "glCompressedTexSubImage3DARB");
    context->GetCompressedTexImageARB = (PFNGLGETCOMPRESSEDTEXIMAGEARBPROC) load(userptr, "glGetCompressedTexImageARB");
}
static void glad_gl_load_GL_ARB_texture_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_multisample) return;
    context->GetMultisamplefv = (PFNGLGETMULTISAMPLEFVPROC) load(userptr, "glGetMultisamplefv");
    context->SampleMaski = (PFNGLSAMPLEMASKIPROC) load(userptr, "glSampleMaski");
    context->TexImage2DMultisample = (PFNGLTEXIMAGE2DMULTISAMPLEPROC) load(userptr, "glTexImage2DMultisample");
    context->TexImage3DMultisample = (PFNGLTEXIMAGE3DMULTISAMPLEPROC) load(userptr, "glTexImage3DMultisample");
}
static void glad_gl_load_GL_ARB_texture_storage(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_storage) return;
    context->TexStorage1D = (PFNGLTEXSTORAGE1DPROC) load(userptr, "glTexStorage1D");
    context->TexStorage2D = (PFNGLTEXSTORAGE2DPROC) load(userptr, "glTexStorage2D");
    context->TexStorage3D = (PFNGLTEXSTORAGE3DPROC) load(userptr, "glTexStorage3D");
}
static void glad_gl_load_GL_ARB_texture_storage_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_storage_multisample) return;
    context->TexStorage2DMultisample = (PFNGLTEXSTORAGE2DMULTISAMPLEPROC) load(userptr, "glTexStorage2DMultisample");
    context->TexStorage3DMultisample = (PFNGLTEXSTORAGE3DMULTISAMPLEPROC) load(userptr, "glTexStorage3DMultisample");
}
static void glad_gl_load_GL_ARB_texture_view(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_texture_view) return;
    context->TextureView = (PFNGLTEXTUREVIEWPROC) load(userptr, "glTextureView");
}
static void glad_gl_load_GL_ARB_timer_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_timer_query) return;
    context->GetQueryObjecti64v = (PFNGLGETQUERYOBJECTI64VPROC) load(userptr, "glGetQueryObjecti64v");
    context->GetQueryObjectui64v = (PFNGLGETQUERYOBJECTUI64VPROC) load(userptr, "glGetQueryObjectui64v");
    context->QueryCounter = (PFNGLQUERYCOUNTERPROC) load(userptr, "glQueryCounter");
}
static void glad_gl_load_GL_ARB_transform_feedback2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_transform_feedback2) return;
    context->BindTransformFeedback = (PFNGLBINDTRANSFORMFEEDBACKPROC) load(userptr, "glBindTransformFeedback");
    context->DeleteTransformFeedbacks = (PFNGLDELETETRANSFORMFEEDBACKSPROC) load(userptr, "glDeleteTransformFeedbacks");
    context->DrawTransformFeedback = (PFNGLDRAWTRANSFORMFEEDBACKPROC) load(userptr, "glDrawTransformFeedback");
    context->GenTransformFeedbacks = (PFNGLGENTRANSFORMFEEDBACKSPROC) load(userptr, "glGenTransformFeedbacks");
    context->IsTransformFeedback = (PFNGLISTRANSFORMFEEDBACKPROC) load(userptr, "glIsTransformFeedback");
    context->PauseTransformFeedback = (PFNGLPAUSETRANSFORMFEEDBACKPROC) load(userptr, "glPauseTransformFeedback");
    context->ResumeTransformFeedback = (PFNGLRESUMETRANSFORMFEEDBACKPROC) load(userptr, "glResumeTransformFeedback");
}
static void glad_gl_load_GL_ARB_transform_feedback3(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_transform_feedback3) return;
    context->BeginQueryIndexed = (PFNGLBEGINQUERYINDEXEDPROC) load(userptr, "glBeginQueryIndexed");
    context->DrawTransformFeedbackStream = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC) load(userptr, "glDrawTransformFeedbackStream");
    context->EndQueryIndexed = (PFNGLENDQUERYINDEXEDPROC) load(userptr, "glEndQueryIndexed");
    context->GetQueryIndexediv = (PFNGLGETQUERYINDEXEDIVPROC) load(userptr, "glGetQueryIndexediv");
}
static void glad_gl_load_GL_ARB_transform_feedback_instanced(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_transform_feedback_instanced) return;
    context->DrawTransformFeedbackInstanced = (PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC) load(userptr, "glDrawTransformFeedbackInstanced");
    context->DrawTransformFeedbackStreamInstanced = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC) load(userptr, "glDrawTransformFeedbackStreamInstanced");
}
static void glad_gl_load_GL_ARB_transpose_matrix(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_transpose_matrix) return;
    context->LoadTransposeMatrixdARB = (PFNGLLOADTRANSPOSEMATRIXDARBPROC) load(userptr, "glLoadTransposeMatrixdARB");
    context->LoadTransposeMatrixfARB = (PFNGLLOADTRANSPOSEMATRIXFARBPROC) load(userptr, "glLoadTransposeMatrixfARB");
    context->MultTransposeMatrixdARB = (PFNGLMULTTRANSPOSEMATRIXDARBPROC) load(userptr, "glMultTransposeMatrixdARB");
    context->MultTransposeMatrixfARB = (PFNGLMULTTRANSPOSEMATRIXFARBPROC) load(userptr, "glMultTransposeMatrixfARB");
}
static void glad_gl_load_GL_ARB_uniform_buffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_uniform_buffer_object) return;
    context->BindBufferBase = (PFNGLBINDBUFFERBASEPROC) load(userptr, "glBindBufferBase");
    context->BindBufferRange = (PFNGLBINDBUFFERRANGEPROC) load(userptr, "glBindBufferRange");
    context->GetActiveUniformBlockName = (PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC) load(userptr, "glGetActiveUniformBlockName");
    context->GetActiveUniformBlockiv = (PFNGLGETACTIVEUNIFORMBLOCKIVPROC) load(userptr, "glGetActiveUniformBlockiv");
    context->GetActiveUniformName = (PFNGLGETACTIVEUNIFORMNAMEPROC) load(userptr, "glGetActiveUniformName");
    context->GetActiveUniformsiv = (PFNGLGETACTIVEUNIFORMSIVPROC) load(userptr, "glGetActiveUniformsiv");
    context->GetIntegeri_v = (PFNGLGETINTEGERI_VPROC) load(userptr, "glGetIntegeri_v");
    context->GetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC) load(userptr, "glGetUniformBlockIndex");
    context->GetUniformIndices = (PFNGLGETUNIFORMINDICESPROC) load(userptr, "glGetUniformIndices");
    context->UniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC) load(userptr, "glUniformBlockBinding");
}
static void glad_gl_load_GL_ARB_vertex_array_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_array_object) return;
    context->BindVertexArray = (PFNGLBINDVERTEXARRAYPROC) load(userptr, "glBindVertexArray");
    context->DeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC) load(userptr, "glDeleteVertexArrays");
    context->GenVertexArrays = (PFNGLGENVERTEXARRAYSPROC) load(userptr, "glGenVertexArrays");
    context->IsVertexArray = (PFNGLISVERTEXARRAYPROC) load(userptr, "glIsVertexArray");
}
static void glad_gl_load_GL_ARB_vertex_attrib_64bit(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_attrib_64bit) return;
    context->GetVertexAttribLdv = (PFNGLGETVERTEXATTRIBLDVPROC) load(userptr, "glGetVertexAttribLdv");
    context->VertexAttribL1d = (PFNGLVERTEXATTRIBL1DPROC) load(userptr, "glVertexAttribL1d");
    context->VertexAttribL1dv = (PFNGLVERTEXATTRIBL1DVPROC) load(userptr, "glVertexAttribL1dv");
    context->VertexAttribL2d = (PFNGLVERTEXATTRIBL2DPROC) load(userptr, "glVertexAttribL2d");
    context->VertexAttribL2dv = (PFNGLVERTEXATTRIBL2DVPROC) load(userptr, "glVertexAttribL2dv");
    context->VertexAttribL3d = (PFNGLVERTEXATTRIBL3DPROC) load(userptr, "glVertexAttribL3d");
    context->VertexAttribL3dv = (PFNGLVERTEXATTRIBL3DVPROC) load(userptr, "glVertexAttribL3dv");
    context->VertexAttribL4d = (PFNGLVERTEXATTRIBL4DPROC) load(userptr, "glVertexAttribL4d");
    context->VertexAttribL4dv = (PFNGLVERTEXATTRIBL4DVPROC) load(userptr, "glVertexAttribL4dv");
    context->VertexAttribLPointer = (PFNGLVERTEXATTRIBLPOINTERPROC) load(userptr, "glVertexAttribLPointer");
}
static void glad_gl_load_GL_ARB_vertex_attrib_binding(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_attrib_binding) return;
    context->BindVertexBuffer = (PFNGLBINDVERTEXBUFFERPROC) load(userptr, "glBindVertexBuffer");
    context->VertexAttribBinding = (PFNGLVERTEXATTRIBBINDINGPROC) load(userptr, "glVertexAttribBinding");
    context->VertexAttribFormat = (PFNGLVERTEXATTRIBFORMATPROC) load(userptr, "glVertexAttribFormat");
    context->VertexAttribIFormat = (PFNGLVERTEXATTRIBIFORMATPROC) load(userptr, "glVertexAttribIFormat");
    context->VertexAttribLFormat = (PFNGLVERTEXATTRIBLFORMATPROC) load(userptr, "glVertexAttribLFormat");
    context->VertexBindingDivisor = (PFNGLVERTEXBINDINGDIVISORPROC) load(userptr, "glVertexBindingDivisor");
}
static void glad_gl_load_GL_ARB_vertex_blend(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_blend) return;
    context->VertexBlendARB = (PFNGLVERTEXBLENDARBPROC) load(userptr, "glVertexBlendARB");
    context->WeightPointerARB = (PFNGLWEIGHTPOINTERARBPROC) load(userptr, "glWeightPointerARB");
    context->WeightbvARB = (PFNGLWEIGHTBVARBPROC) load(userptr, "glWeightbvARB");
    context->WeightdvARB = (PFNGLWEIGHTDVARBPROC) load(userptr, "glWeightdvARB");
    context->WeightfvARB = (PFNGLWEIGHTFVARBPROC) load(userptr, "glWeightfvARB");
    context->WeightivARB = (PFNGLWEIGHTIVARBPROC) load(userptr, "glWeightivARB");
    context->WeightsvARB = (PFNGLWEIGHTSVARBPROC) load(userptr, "glWeightsvARB");
    context->WeightubvARB = (PFNGLWEIGHTUBVARBPROC) load(userptr, "glWeightubvARB");
    context->WeightuivARB = (PFNGLWEIGHTUIVARBPROC) load(userptr, "glWeightuivARB");
    context->WeightusvARB = (PFNGLWEIGHTUSVARBPROC) load(userptr, "glWeightusvARB");
}
static void glad_gl_load_GL_ARB_vertex_buffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_buffer_object) return;
    context->BindBufferARB = (PFNGLBINDBUFFERARBPROC) load(userptr, "glBindBufferARB");
    context->BufferDataARB = (PFNGLBUFFERDATAARBPROC) load(userptr, "glBufferDataARB");
    context->BufferSubDataARB = (PFNGLBUFFERSUBDATAARBPROC) load(userptr, "glBufferSubDataARB");
    context->DeleteBuffersARB = (PFNGLDELETEBUFFERSARBPROC) load(userptr, "glDeleteBuffersARB");
    context->GenBuffersARB = (PFNGLGENBUFFERSARBPROC) load(userptr, "glGenBuffersARB");
    context->GetBufferParameterivARB = (PFNGLGETBUFFERPARAMETERIVARBPROC) load(userptr, "glGetBufferParameterivARB");
    context->GetBufferPointervARB = (PFNGLGETBUFFERPOINTERVARBPROC) load(userptr, "glGetBufferPointervARB");
    context->GetBufferSubDataARB = (PFNGLGETBUFFERSUBDATAARBPROC) load(userptr, "glGetBufferSubDataARB");
    context->IsBufferARB = (PFNGLISBUFFERARBPROC) load(userptr, "glIsBufferARB");
    context->MapBufferARB = (PFNGLMAPBUFFERARBPROC) load(userptr, "glMapBufferARB");
    context->UnmapBufferARB = (PFNGLUNMAPBUFFERARBPROC) load(userptr, "glUnmapBufferARB");
}
static void glad_gl_load_GL_ARB_vertex_program(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_program) return;
    context->BindProgramARB = (PFNGLBINDPROGRAMARBPROC) load(userptr, "glBindProgramARB");
    context->DeleteProgramsARB = (PFNGLDELETEPROGRAMSARBPROC) load(userptr, "glDeleteProgramsARB");
    context->DisableVertexAttribArrayARB = (PFNGLDISABLEVERTEXATTRIBARRAYARBPROC) load(userptr, "glDisableVertexAttribArrayARB");
    context->EnableVertexAttribArrayARB = (PFNGLENABLEVERTEXATTRIBARRAYARBPROC) load(userptr, "glEnableVertexAttribArrayARB");
    context->GenProgramsARB = (PFNGLGENPROGRAMSARBPROC) load(userptr, "glGenProgramsARB");
    context->GetProgramEnvParameterdvARB = (PFNGLGETPROGRAMENVPARAMETERDVARBPROC) load(userptr, "glGetProgramEnvParameterdvARB");
    context->GetProgramEnvParameterfvARB = (PFNGLGETPROGRAMENVPARAMETERFVARBPROC) load(userptr, "glGetProgramEnvParameterfvARB");
    context->GetProgramLocalParameterdvARB = (PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC) load(userptr, "glGetProgramLocalParameterdvARB");
    context->GetProgramLocalParameterfvARB = (PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC) load(userptr, "glGetProgramLocalParameterfvARB");
    context->GetProgramStringARB = (PFNGLGETPROGRAMSTRINGARBPROC) load(userptr, "glGetProgramStringARB");
    context->GetProgramivARB = (PFNGLGETPROGRAMIVARBPROC) load(userptr, "glGetProgramivARB");
    context->GetVertexAttribPointervARB = (PFNGLGETVERTEXATTRIBPOINTERVARBPROC) load(userptr, "glGetVertexAttribPointervARB");
    context->GetVertexAttribdvARB = (PFNGLGETVERTEXATTRIBDVARBPROC) load(userptr, "glGetVertexAttribdvARB");
    context->GetVertexAttribfvARB = (PFNGLGETVERTEXATTRIBFVARBPROC) load(userptr, "glGetVertexAttribfvARB");
    context->GetVertexAttribivARB = (PFNGLGETVERTEXATTRIBIVARBPROC) load(userptr, "glGetVertexAttribivARB");
    context->IsProgramARB = (PFNGLISPROGRAMARBPROC) load(userptr, "glIsProgramARB");
    context->ProgramEnvParameter4dARB = (PFNGLPROGRAMENVPARAMETER4DARBPROC) load(userptr, "glProgramEnvParameter4dARB");
    context->ProgramEnvParameter4dvARB = (PFNGLPROGRAMENVPARAMETER4DVARBPROC) load(userptr, "glProgramEnvParameter4dvARB");
    context->ProgramEnvParameter4fARB = (PFNGLPROGRAMENVPARAMETER4FARBPROC) load(userptr, "glProgramEnvParameter4fARB");
    context->ProgramEnvParameter4fvARB = (PFNGLPROGRAMENVPARAMETER4FVARBPROC) load(userptr, "glProgramEnvParameter4fvARB");
    context->ProgramLocalParameter4dARB = (PFNGLPROGRAMLOCALPARAMETER4DARBPROC) load(userptr, "glProgramLocalParameter4dARB");
    context->ProgramLocalParameter4dvARB = (PFNGLPROGRAMLOCALPARAMETER4DVARBPROC) load(userptr, "glProgramLocalParameter4dvARB");
    context->ProgramLocalParameter4fARB = (PFNGLPROGRAMLOCALPARAMETER4FARBPROC) load(userptr, "glProgramLocalParameter4fARB");
    context->ProgramLocalParameter4fvARB = (PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) load(userptr, "glProgramLocalParameter4fvARB");
    context->ProgramStringARB = (PFNGLPROGRAMSTRINGARBPROC) load(userptr, "glProgramStringARB");
    context->VertexAttrib1dARB = (PFNGLVERTEXATTRIB1DARBPROC) load(userptr, "glVertexAttrib1dARB");
    context->VertexAttrib1dvARB = (PFNGLVERTEXATTRIB1DVARBPROC) load(userptr, "glVertexAttrib1dvARB");
    context->VertexAttrib1fARB = (PFNGLVERTEXATTRIB1FARBPROC) load(userptr, "glVertexAttrib1fARB");
    context->VertexAttrib1fvARB = (PFNGLVERTEXATTRIB1FVARBPROC) load(userptr, "glVertexAttrib1fvARB");
    context->VertexAttrib1sARB = (PFNGLVERTEXATTRIB1SARBPROC) load(userptr, "glVertexAttrib1sARB");
    context->VertexAttrib1svARB = (PFNGLVERTEXATTRIB1SVARBPROC) load(userptr, "glVertexAttrib1svARB");
    context->VertexAttrib2dARB = (PFNGLVERTEXATTRIB2DARBPROC) load(userptr, "glVertexAttrib2dARB");
    context->VertexAttrib2dvARB = (PFNGLVERTEXATTRIB2DVARBPROC) load(userptr, "glVertexAttrib2dvARB");
    context->VertexAttrib2fARB = (PFNGLVERTEXATTRIB2FARBPROC) load(userptr, "glVertexAttrib2fARB");
    context->VertexAttrib2fvARB = (PFNGLVERTEXATTRIB2FVARBPROC) load(userptr, "glVertexAttrib2fvARB");
    context->VertexAttrib2sARB = (PFNGLVERTEXATTRIB2SARBPROC) load(userptr, "glVertexAttrib2sARB");
    context->VertexAttrib2svARB = (PFNGLVERTEXATTRIB2SVARBPROC) load(userptr, "glVertexAttrib2svARB");
    context->VertexAttrib3dARB = (PFNGLVERTEXATTRIB3DARBPROC) load(userptr, "glVertexAttrib3dARB");
    context->VertexAttrib3dvARB = (PFNGLVERTEXATTRIB3DVARBPROC) load(userptr, "glVertexAttrib3dvARB");
    context->VertexAttrib3fARB = (PFNGLVERTEXATTRIB3FARBPROC) load(userptr, "glVertexAttrib3fARB");
    context->VertexAttrib3fvARB = (PFNGLVERTEXATTRIB3FVARBPROC) load(userptr, "glVertexAttrib3fvARB");
    context->VertexAttrib3sARB = (PFNGLVERTEXATTRIB3SARBPROC) load(userptr, "glVertexAttrib3sARB");
    context->VertexAttrib3svARB = (PFNGLVERTEXATTRIB3SVARBPROC) load(userptr, "glVertexAttrib3svARB");
    context->VertexAttrib4NbvARB = (PFNGLVERTEXATTRIB4NBVARBPROC) load(userptr, "glVertexAttrib4NbvARB");
    context->VertexAttrib4NivARB = (PFNGLVERTEXATTRIB4NIVARBPROC) load(userptr, "glVertexAttrib4NivARB");
    context->VertexAttrib4NsvARB = (PFNGLVERTEXATTRIB4NSVARBPROC) load(userptr, "glVertexAttrib4NsvARB");
    context->VertexAttrib4NubARB = (PFNGLVERTEXATTRIB4NUBARBPROC) load(userptr, "glVertexAttrib4NubARB");
    context->VertexAttrib4NubvARB = (PFNGLVERTEXATTRIB4NUBVARBPROC) load(userptr, "glVertexAttrib4NubvARB");
    context->VertexAttrib4NuivARB = (PFNGLVERTEXATTRIB4NUIVARBPROC) load(userptr, "glVertexAttrib4NuivARB");
    context->VertexAttrib4NusvARB = (PFNGLVERTEXATTRIB4NUSVARBPROC) load(userptr, "glVertexAttrib4NusvARB");
    context->VertexAttrib4bvARB = (PFNGLVERTEXATTRIB4BVARBPROC) load(userptr, "glVertexAttrib4bvARB");
    context->VertexAttrib4dARB = (PFNGLVERTEXATTRIB4DARBPROC) load(userptr, "glVertexAttrib4dARB");
    context->VertexAttrib4dvARB = (PFNGLVERTEXATTRIB4DVARBPROC) load(userptr, "glVertexAttrib4dvARB");
    context->VertexAttrib4fARB = (PFNGLVERTEXATTRIB4FARBPROC) load(userptr, "glVertexAttrib4fARB");
    context->VertexAttrib4fvARB = (PFNGLVERTEXATTRIB4FVARBPROC) load(userptr, "glVertexAttrib4fvARB");
    context->VertexAttrib4ivARB = (PFNGLVERTEXATTRIB4IVARBPROC) load(userptr, "glVertexAttrib4ivARB");
    context->VertexAttrib4sARB = (PFNGLVERTEXATTRIB4SARBPROC) load(userptr, "glVertexAttrib4sARB");
    context->VertexAttrib4svARB = (PFNGLVERTEXATTRIB4SVARBPROC) load(userptr, "glVertexAttrib4svARB");
    context->VertexAttrib4ubvARB = (PFNGLVERTEXATTRIB4UBVARBPROC) load(userptr, "glVertexAttrib4ubvARB");
    context->VertexAttrib4uivARB = (PFNGLVERTEXATTRIB4UIVARBPROC) load(userptr, "glVertexAttrib4uivARB");
    context->VertexAttrib4usvARB = (PFNGLVERTEXATTRIB4USVARBPROC) load(userptr, "glVertexAttrib4usvARB");
    context->VertexAttribPointerARB = (PFNGLVERTEXATTRIBPOINTERARBPROC) load(userptr, "glVertexAttribPointerARB");
}
static void glad_gl_load_GL_ARB_vertex_shader(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_shader) return;
    context->BindAttribLocationARB = (PFNGLBINDATTRIBLOCATIONARBPROC) load(userptr, "glBindAttribLocationARB");
    context->DisableVertexAttribArrayARB = (PFNGLDISABLEVERTEXATTRIBARRAYARBPROC) load(userptr, "glDisableVertexAttribArrayARB");
    context->EnableVertexAttribArrayARB = (PFNGLENABLEVERTEXATTRIBARRAYARBPROC) load(userptr, "glEnableVertexAttribArrayARB");
    context->GetActiveAttribARB = (PFNGLGETACTIVEATTRIBARBPROC) load(userptr, "glGetActiveAttribARB");
    context->GetAttribLocationARB = (PFNGLGETATTRIBLOCATIONARBPROC) load(userptr, "glGetAttribLocationARB");
    context->GetVertexAttribPointervARB = (PFNGLGETVERTEXATTRIBPOINTERVARBPROC) load(userptr, "glGetVertexAttribPointervARB");
    context->GetVertexAttribdvARB = (PFNGLGETVERTEXATTRIBDVARBPROC) load(userptr, "glGetVertexAttribdvARB");
    context->GetVertexAttribfvARB = (PFNGLGETVERTEXATTRIBFVARBPROC) load(userptr, "glGetVertexAttribfvARB");
    context->GetVertexAttribivARB = (PFNGLGETVERTEXATTRIBIVARBPROC) load(userptr, "glGetVertexAttribivARB");
    context->VertexAttrib1dARB = (PFNGLVERTEXATTRIB1DARBPROC) load(userptr, "glVertexAttrib1dARB");
    context->VertexAttrib1dvARB = (PFNGLVERTEXATTRIB1DVARBPROC) load(userptr, "glVertexAttrib1dvARB");
    context->VertexAttrib1fARB = (PFNGLVERTEXATTRIB1FARBPROC) load(userptr, "glVertexAttrib1fARB");
    context->VertexAttrib1fvARB = (PFNGLVERTEXATTRIB1FVARBPROC) load(userptr, "glVertexAttrib1fvARB");
    context->VertexAttrib1sARB = (PFNGLVERTEXATTRIB1SARBPROC) load(userptr, "glVertexAttrib1sARB");
    context->VertexAttrib1svARB = (PFNGLVERTEXATTRIB1SVARBPROC) load(userptr, "glVertexAttrib1svARB");
    context->VertexAttrib2dARB = (PFNGLVERTEXATTRIB2DARBPROC) load(userptr, "glVertexAttrib2dARB");
    context->VertexAttrib2dvARB = (PFNGLVERTEXATTRIB2DVARBPROC) load(userptr, "glVertexAttrib2dvARB");
    context->VertexAttrib2fARB = (PFNGLVERTEXATTRIB2FARBPROC) load(userptr, "glVertexAttrib2fARB");
    context->VertexAttrib2fvARB = (PFNGLVERTEXATTRIB2FVARBPROC) load(userptr, "glVertexAttrib2fvARB");
    context->VertexAttrib2sARB = (PFNGLVERTEXATTRIB2SARBPROC) load(userptr, "glVertexAttrib2sARB");
    context->VertexAttrib2svARB = (PFNGLVERTEXATTRIB2SVARBPROC) load(userptr, "glVertexAttrib2svARB");
    context->VertexAttrib3dARB = (PFNGLVERTEXATTRIB3DARBPROC) load(userptr, "glVertexAttrib3dARB");
    context->VertexAttrib3dvARB = (PFNGLVERTEXATTRIB3DVARBPROC) load(userptr, "glVertexAttrib3dvARB");
    context->VertexAttrib3fARB = (PFNGLVERTEXATTRIB3FARBPROC) load(userptr, "glVertexAttrib3fARB");
    context->VertexAttrib3fvARB = (PFNGLVERTEXATTRIB3FVARBPROC) load(userptr, "glVertexAttrib3fvARB");
    context->VertexAttrib3sARB = (PFNGLVERTEXATTRIB3SARBPROC) load(userptr, "glVertexAttrib3sARB");
    context->VertexAttrib3svARB = (PFNGLVERTEXATTRIB3SVARBPROC) load(userptr, "glVertexAttrib3svARB");
    context->VertexAttrib4NbvARB = (PFNGLVERTEXATTRIB4NBVARBPROC) load(userptr, "glVertexAttrib4NbvARB");
    context->VertexAttrib4NivARB = (PFNGLVERTEXATTRIB4NIVARBPROC) load(userptr, "glVertexAttrib4NivARB");
    context->VertexAttrib4NsvARB = (PFNGLVERTEXATTRIB4NSVARBPROC) load(userptr, "glVertexAttrib4NsvARB");
    context->VertexAttrib4NubARB = (PFNGLVERTEXATTRIB4NUBARBPROC) load(userptr, "glVertexAttrib4NubARB");
    context->VertexAttrib4NubvARB = (PFNGLVERTEXATTRIB4NUBVARBPROC) load(userptr, "glVertexAttrib4NubvARB");
    context->VertexAttrib4NuivARB = (PFNGLVERTEXATTRIB4NUIVARBPROC) load(userptr, "glVertexAttrib4NuivARB");
    context->VertexAttrib4NusvARB = (PFNGLVERTEXATTRIB4NUSVARBPROC) load(userptr, "glVertexAttrib4NusvARB");
    context->VertexAttrib4bvARB = (PFNGLVERTEXATTRIB4BVARBPROC) load(userptr, "glVertexAttrib4bvARB");
    context->VertexAttrib4dARB = (PFNGLVERTEXATTRIB4DARBPROC) load(userptr, "glVertexAttrib4dARB");
    context->VertexAttrib4dvARB = (PFNGLVERTEXATTRIB4DVARBPROC) load(userptr, "glVertexAttrib4dvARB");
    context->VertexAttrib4fARB = (PFNGLVERTEXATTRIB4FARBPROC) load(userptr, "glVertexAttrib4fARB");
    context->VertexAttrib4fvARB = (PFNGLVERTEXATTRIB4FVARBPROC) load(userptr, "glVertexAttrib4fvARB");
    context->VertexAttrib4ivARB = (PFNGLVERTEXATTRIB4IVARBPROC) load(userptr, "glVertexAttrib4ivARB");
    context->VertexAttrib4sARB = (PFNGLVERTEXATTRIB4SARBPROC) load(userptr, "glVertexAttrib4sARB");
    context->VertexAttrib4svARB = (PFNGLVERTEXATTRIB4SVARBPROC) load(userptr, "glVertexAttrib4svARB");
    context->VertexAttrib4ubvARB = (PFNGLVERTEXATTRIB4UBVARBPROC) load(userptr, "glVertexAttrib4ubvARB");
    context->VertexAttrib4uivARB = (PFNGLVERTEXATTRIB4UIVARBPROC) load(userptr, "glVertexAttrib4uivARB");
    context->VertexAttrib4usvARB = (PFNGLVERTEXATTRIB4USVARBPROC) load(userptr, "glVertexAttrib4usvARB");
    context->VertexAttribPointerARB = (PFNGLVERTEXATTRIBPOINTERARBPROC) load(userptr, "glVertexAttribPointerARB");
}
static void glad_gl_load_GL_ARB_vertex_type_2_10_10_10_rev(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_vertex_type_2_10_10_10_rev) return;
    context->ColorP3ui = (PFNGLCOLORP3UIPROC) load(userptr, "glColorP3ui");
    context->ColorP3uiv = (PFNGLCOLORP3UIVPROC) load(userptr, "glColorP3uiv");
    context->ColorP4ui = (PFNGLCOLORP4UIPROC) load(userptr, "glColorP4ui");
    context->ColorP4uiv = (PFNGLCOLORP4UIVPROC) load(userptr, "glColorP4uiv");
    context->MultiTexCoordP1ui = (PFNGLMULTITEXCOORDP1UIPROC) load(userptr, "glMultiTexCoordP1ui");
    context->MultiTexCoordP1uiv = (PFNGLMULTITEXCOORDP1UIVPROC) load(userptr, "glMultiTexCoordP1uiv");
    context->MultiTexCoordP2ui = (PFNGLMULTITEXCOORDP2UIPROC) load(userptr, "glMultiTexCoordP2ui");
    context->MultiTexCoordP2uiv = (PFNGLMULTITEXCOORDP2UIVPROC) load(userptr, "glMultiTexCoordP2uiv");
    context->MultiTexCoordP3ui = (PFNGLMULTITEXCOORDP3UIPROC) load(userptr, "glMultiTexCoordP3ui");
    context->MultiTexCoordP3uiv = (PFNGLMULTITEXCOORDP3UIVPROC) load(userptr, "glMultiTexCoordP3uiv");
    context->MultiTexCoordP4ui = (PFNGLMULTITEXCOORDP4UIPROC) load(userptr, "glMultiTexCoordP4ui");
    context->MultiTexCoordP4uiv = (PFNGLMULTITEXCOORDP4UIVPROC) load(userptr, "glMultiTexCoordP4uiv");
    context->NormalP3ui = (PFNGLNORMALP3UIPROC) load(userptr, "glNormalP3ui");
    context->NormalP3uiv = (PFNGLNORMALP3UIVPROC) load(userptr, "glNormalP3uiv");
    context->SecondaryColorP3ui = (PFNGLSECONDARYCOLORP3UIPROC) load(userptr, "glSecondaryColorP3ui");
    context->SecondaryColorP3uiv = (PFNGLSECONDARYCOLORP3UIVPROC) load(userptr, "glSecondaryColorP3uiv");
    context->TexCoordP1ui = (PFNGLTEXCOORDP1UIPROC) load(userptr, "glTexCoordP1ui");
    context->TexCoordP1uiv = (PFNGLTEXCOORDP1UIVPROC) load(userptr, "glTexCoordP1uiv");
    context->TexCoordP2ui = (PFNGLTEXCOORDP2UIPROC) load(userptr, "glTexCoordP2ui");
    context->TexCoordP2uiv = (PFNGLTEXCOORDP2UIVPROC) load(userptr, "glTexCoordP2uiv");
    context->TexCoordP3ui = (PFNGLTEXCOORDP3UIPROC) load(userptr, "glTexCoordP3ui");
    context->TexCoordP3uiv = (PFNGLTEXCOORDP3UIVPROC) load(userptr, "glTexCoordP3uiv");
    context->TexCoordP4ui = (PFNGLTEXCOORDP4UIPROC) load(userptr, "glTexCoordP4ui");
    context->TexCoordP4uiv = (PFNGLTEXCOORDP4UIVPROC) load(userptr, "glTexCoordP4uiv");
    context->VertexAttribP1ui = (PFNGLVERTEXATTRIBP1UIPROC) load(userptr, "glVertexAttribP1ui");
    context->VertexAttribP1uiv = (PFNGLVERTEXATTRIBP1UIVPROC) load(userptr, "glVertexAttribP1uiv");
    context->VertexAttribP2ui = (PFNGLVERTEXATTRIBP2UIPROC) load(userptr, "glVertexAttribP2ui");
    context->VertexAttribP2uiv = (PFNGLVERTEXATTRIBP2UIVPROC) load(userptr, "glVertexAttribP2uiv");
    context->VertexAttribP3ui = (PFNGLVERTEXATTRIBP3UIPROC) load(userptr, "glVertexAttribP3ui");
    context->VertexAttribP3uiv = (PFNGLVERTEXATTRIBP3UIVPROC) load(userptr, "glVertexAttribP3uiv");
    context->VertexAttribP4ui = (PFNGLVERTEXATTRIBP4UIPROC) load(userptr, "glVertexAttribP4ui");
    context->VertexAttribP4uiv = (PFNGLVERTEXATTRIBP4UIVPROC) load(userptr, "glVertexAttribP4uiv");
    context->VertexP2ui = (PFNGLVERTEXP2UIPROC) load(userptr, "glVertexP2ui");
    context->VertexP2uiv = (PFNGLVERTEXP2UIVPROC) load(userptr, "glVertexP2uiv");
    context->VertexP3ui = (PFNGLVERTEXP3UIPROC) load(userptr, "glVertexP3ui");
    context->VertexP3uiv = (PFNGLVERTEXP3UIVPROC) load(userptr, "glVertexP3uiv");
    context->VertexP4ui = (PFNGLVERTEXP4UIPROC) load(userptr, "glVertexP4ui");
    context->VertexP4uiv = (PFNGLVERTEXP4UIVPROC) load(userptr, "glVertexP4uiv");
}
static void glad_gl_load_GL_ARB_viewport_array(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_viewport_array) return;
    context->DepthRangeArraydvNV = (PFNGLDEPTHRANGEARRAYDVNVPROC) load(userptr, "glDepthRangeArraydvNV");
    context->DepthRangeArrayv = (PFNGLDEPTHRANGEARRAYVPROC) load(userptr, "glDepthRangeArrayv");
    context->DepthRangeIndexed = (PFNGLDEPTHRANGEINDEXEDPROC) load(userptr, "glDepthRangeIndexed");
    context->DepthRangeIndexeddNV = (PFNGLDEPTHRANGEINDEXEDDNVPROC) load(userptr, "glDepthRangeIndexeddNV");
    context->GetDoublei_v = (PFNGLGETDOUBLEI_VPROC) load(userptr, "glGetDoublei_v");
    context->GetFloati_v = (PFNGLGETFLOATI_VPROC) load(userptr, "glGetFloati_v");
    context->ScissorArrayv = (PFNGLSCISSORARRAYVPROC) load(userptr, "glScissorArrayv");
    context->ScissorIndexed = (PFNGLSCISSORINDEXEDPROC) load(userptr, "glScissorIndexed");
    context->ScissorIndexedv = (PFNGLSCISSORINDEXEDVPROC) load(userptr, "glScissorIndexedv");
    context->ViewportArrayv = (PFNGLVIEWPORTARRAYVPROC) load(userptr, "glViewportArrayv");
    context->ViewportIndexedf = (PFNGLVIEWPORTINDEXEDFPROC) load(userptr, "glViewportIndexedf");
    context->ViewportIndexedfv = (PFNGLVIEWPORTINDEXEDFVPROC) load(userptr, "glViewportIndexedfv");
}
static void glad_gl_load_GL_ARB_window_pos(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ARB_window_pos) return;
    context->WindowPos2dARB = (PFNGLWINDOWPOS2DARBPROC) load(userptr, "glWindowPos2dARB");
    context->WindowPos2dvARB = (PFNGLWINDOWPOS2DVARBPROC) load(userptr, "glWindowPos2dvARB");
    context->WindowPos2fARB = (PFNGLWINDOWPOS2FARBPROC) load(userptr, "glWindowPos2fARB");
    context->WindowPos2fvARB = (PFNGLWINDOWPOS2FVARBPROC) load(userptr, "glWindowPos2fvARB");
    context->WindowPos2iARB = (PFNGLWINDOWPOS2IARBPROC) load(userptr, "glWindowPos2iARB");
    context->WindowPos2ivARB = (PFNGLWINDOWPOS2IVARBPROC) load(userptr, "glWindowPos2ivARB");
    context->WindowPos2sARB = (PFNGLWINDOWPOS2SARBPROC) load(userptr, "glWindowPos2sARB");
    context->WindowPos2svARB = (PFNGLWINDOWPOS2SVARBPROC) load(userptr, "glWindowPos2svARB");
    context->WindowPos3dARB = (PFNGLWINDOWPOS3DARBPROC) load(userptr, "glWindowPos3dARB");
    context->WindowPos3dvARB = (PFNGLWINDOWPOS3DVARBPROC) load(userptr, "glWindowPos3dvARB");
    context->WindowPos3fARB = (PFNGLWINDOWPOS3FARBPROC) load(userptr, "glWindowPos3fARB");
    context->WindowPos3fvARB = (PFNGLWINDOWPOS3FVARBPROC) load(userptr, "glWindowPos3fvARB");
    context->WindowPos3iARB = (PFNGLWINDOWPOS3IARBPROC) load(userptr, "glWindowPos3iARB");
    context->WindowPos3ivARB = (PFNGLWINDOWPOS3IVARBPROC) load(userptr, "glWindowPos3ivARB");
    context->WindowPos3sARB = (PFNGLWINDOWPOS3SARBPROC) load(userptr, "glWindowPos3sARB");
    context->WindowPos3svARB = (PFNGLWINDOWPOS3SVARBPROC) load(userptr, "glWindowPos3svARB");
}
static void glad_gl_load_GL_ATI_draw_buffers(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_draw_buffers) return;
    context->DrawBuffersATI = (PFNGLDRAWBUFFERSATIPROC) load(userptr, "glDrawBuffersATI");
}
static void glad_gl_load_GL_ATI_element_array(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_element_array) return;
    context->DrawElementArrayATI = (PFNGLDRAWELEMENTARRAYATIPROC) load(userptr, "glDrawElementArrayATI");
    context->DrawRangeElementArrayATI = (PFNGLDRAWRANGEELEMENTARRAYATIPROC) load(userptr, "glDrawRangeElementArrayATI");
    context->ElementPointerATI = (PFNGLELEMENTPOINTERATIPROC) load(userptr, "glElementPointerATI");
}
static void glad_gl_load_GL_ATI_envmap_bumpmap(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_envmap_bumpmap) return;
    context->GetTexBumpParameterfvATI = (PFNGLGETTEXBUMPPARAMETERFVATIPROC) load(userptr, "glGetTexBumpParameterfvATI");
    context->GetTexBumpParameterivATI = (PFNGLGETTEXBUMPPARAMETERIVATIPROC) load(userptr, "glGetTexBumpParameterivATI");
    context->TexBumpParameterfvATI = (PFNGLTEXBUMPPARAMETERFVATIPROC) load(userptr, "glTexBumpParameterfvATI");
    context->TexBumpParameterivATI = (PFNGLTEXBUMPPARAMETERIVATIPROC) load(userptr, "glTexBumpParameterivATI");
}
static void glad_gl_load_GL_ATI_fragment_shader(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_fragment_shader) return;
    context->AlphaFragmentOp1ATI = (PFNGLALPHAFRAGMENTOP1ATIPROC) load(userptr, "glAlphaFragmentOp1ATI");
    context->AlphaFragmentOp2ATI = (PFNGLALPHAFRAGMENTOP2ATIPROC) load(userptr, "glAlphaFragmentOp2ATI");
    context->AlphaFragmentOp3ATI = (PFNGLALPHAFRAGMENTOP3ATIPROC) load(userptr, "glAlphaFragmentOp3ATI");
    context->BeginFragmentShaderATI = (PFNGLBEGINFRAGMENTSHADERATIPROC) load(userptr, "glBeginFragmentShaderATI");
    context->BindFragmentShaderATI = (PFNGLBINDFRAGMENTSHADERATIPROC) load(userptr, "glBindFragmentShaderATI");
    context->ColorFragmentOp1ATI = (PFNGLCOLORFRAGMENTOP1ATIPROC) load(userptr, "glColorFragmentOp1ATI");
    context->ColorFragmentOp2ATI = (PFNGLCOLORFRAGMENTOP2ATIPROC) load(userptr, "glColorFragmentOp2ATI");
    context->ColorFragmentOp3ATI = (PFNGLCOLORFRAGMENTOP3ATIPROC) load(userptr, "glColorFragmentOp3ATI");
    context->DeleteFragmentShaderATI = (PFNGLDELETEFRAGMENTSHADERATIPROC) load(userptr, "glDeleteFragmentShaderATI");
    context->EndFragmentShaderATI = (PFNGLENDFRAGMENTSHADERATIPROC) load(userptr, "glEndFragmentShaderATI");
    context->GenFragmentShadersATI = (PFNGLGENFRAGMENTSHADERSATIPROC) load(userptr, "glGenFragmentShadersATI");
    context->PassTexCoordATI = (PFNGLPASSTEXCOORDATIPROC) load(userptr, "glPassTexCoordATI");
    context->SampleMapATI = (PFNGLSAMPLEMAPATIPROC) load(userptr, "glSampleMapATI");
    context->SetFragmentShaderConstantATI = (PFNGLSETFRAGMENTSHADERCONSTANTATIPROC) load(userptr, "glSetFragmentShaderConstantATI");
}
static void glad_gl_load_GL_ATI_map_object_buffer(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_map_object_buffer) return;
    context->MapObjectBufferATI = (PFNGLMAPOBJECTBUFFERATIPROC) load(userptr, "glMapObjectBufferATI");
    context->UnmapObjectBufferATI = (PFNGLUNMAPOBJECTBUFFERATIPROC) load(userptr, "glUnmapObjectBufferATI");
}
static void glad_gl_load_GL_ATI_pn_triangles(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_pn_triangles) return;
    context->PNTrianglesfATI = (PFNGLPNTRIANGLESFATIPROC) load(userptr, "glPNTrianglesfATI");
    context->PNTrianglesiATI = (PFNGLPNTRIANGLESIATIPROC) load(userptr, "glPNTrianglesiATI");
}
static void glad_gl_load_GL_ATI_separate_stencil(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_separate_stencil) return;
    context->StencilFuncSeparateATI = (PFNGLSTENCILFUNCSEPARATEATIPROC) load(userptr, "glStencilFuncSeparateATI");
    context->StencilOpSeparateATI = (PFNGLSTENCILOPSEPARATEATIPROC) load(userptr, "glStencilOpSeparateATI");
}
static void glad_gl_load_GL_ATI_vertex_array_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_vertex_array_object) return;
    context->ArrayObjectATI = (PFNGLARRAYOBJECTATIPROC) load(userptr, "glArrayObjectATI");
    context->FreeObjectBufferATI = (PFNGLFREEOBJECTBUFFERATIPROC) load(userptr, "glFreeObjectBufferATI");
    context->GetArrayObjectfvATI = (PFNGLGETARRAYOBJECTFVATIPROC) load(userptr, "glGetArrayObjectfvATI");
    context->GetArrayObjectivATI = (PFNGLGETARRAYOBJECTIVATIPROC) load(userptr, "glGetArrayObjectivATI");
    context->GetObjectBufferfvATI = (PFNGLGETOBJECTBUFFERFVATIPROC) load(userptr, "glGetObjectBufferfvATI");
    context->GetObjectBufferivATI = (PFNGLGETOBJECTBUFFERIVATIPROC) load(userptr, "glGetObjectBufferivATI");
    context->GetVariantArrayObjectfvATI = (PFNGLGETVARIANTARRAYOBJECTFVATIPROC) load(userptr, "glGetVariantArrayObjectfvATI");
    context->GetVariantArrayObjectivATI = (PFNGLGETVARIANTARRAYOBJECTIVATIPROC) load(userptr, "glGetVariantArrayObjectivATI");
    context->IsObjectBufferATI = (PFNGLISOBJECTBUFFERATIPROC) load(userptr, "glIsObjectBufferATI");
    context->NewObjectBufferATI = (PFNGLNEWOBJECTBUFFERATIPROC) load(userptr, "glNewObjectBufferATI");
    context->UpdateObjectBufferATI = (PFNGLUPDATEOBJECTBUFFERATIPROC) load(userptr, "glUpdateObjectBufferATI");
    context->VariantArrayObjectATI = (PFNGLVARIANTARRAYOBJECTATIPROC) load(userptr, "glVariantArrayObjectATI");
}
static void glad_gl_load_GL_ATI_vertex_attrib_array_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_vertex_attrib_array_object) return;
    context->GetVertexAttribArrayObjectfvATI = (PFNGLGETVERTEXATTRIBARRAYOBJECTFVATIPROC) load(userptr, "glGetVertexAttribArrayObjectfvATI");
    context->GetVertexAttribArrayObjectivATI = (PFNGLGETVERTEXATTRIBARRAYOBJECTIVATIPROC) load(userptr, "glGetVertexAttribArrayObjectivATI");
    context->VertexAttribArrayObjectATI = (PFNGLVERTEXATTRIBARRAYOBJECTATIPROC) load(userptr, "glVertexAttribArrayObjectATI");
}
static void glad_gl_load_GL_ATI_vertex_streams(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->ATI_vertex_streams) return;
    context->ClientActiveVertexStreamATI = (PFNGLCLIENTACTIVEVERTEXSTREAMATIPROC) load(userptr, "glClientActiveVertexStreamATI");
    context->NormalStream3bATI = (PFNGLNORMALSTREAM3BATIPROC) load(userptr, "glNormalStream3bATI");
    context->NormalStream3bvATI = (PFNGLNORMALSTREAM3BVATIPROC) load(userptr, "glNormalStream3bvATI");
    context->NormalStream3dATI = (PFNGLNORMALSTREAM3DATIPROC) load(userptr, "glNormalStream3dATI");
    context->NormalStream3dvATI = (PFNGLNORMALSTREAM3DVATIPROC) load(userptr, "glNormalStream3dvATI");
    context->NormalStream3fATI = (PFNGLNORMALSTREAM3FATIPROC) load(userptr, "glNormalStream3fATI");
    context->NormalStream3fvATI = (PFNGLNORMALSTREAM3FVATIPROC) load(userptr, "glNormalStream3fvATI");
    context->NormalStream3iATI = (PFNGLNORMALSTREAM3IATIPROC) load(userptr, "glNormalStream3iATI");
    context->NormalStream3ivATI = (PFNGLNORMALSTREAM3IVATIPROC) load(userptr, "glNormalStream3ivATI");
    context->NormalStream3sATI = (PFNGLNORMALSTREAM3SATIPROC) load(userptr, "glNormalStream3sATI");
    context->NormalStream3svATI = (PFNGLNORMALSTREAM3SVATIPROC) load(userptr, "glNormalStream3svATI");
    context->VertexBlendEnvfATI = (PFNGLVERTEXBLENDENVFATIPROC) load(userptr, "glVertexBlendEnvfATI");
    context->VertexBlendEnviATI = (PFNGLVERTEXBLENDENVIATIPROC) load(userptr, "glVertexBlendEnviATI");
    context->VertexStream1dATI = (PFNGLVERTEXSTREAM1DATIPROC) load(userptr, "glVertexStream1dATI");
    context->VertexStream1dvATI = (PFNGLVERTEXSTREAM1DVATIPROC) load(userptr, "glVertexStream1dvATI");
    context->VertexStream1fATI = (PFNGLVERTEXSTREAM1FATIPROC) load(userptr, "glVertexStream1fATI");
    context->VertexStream1fvATI = (PFNGLVERTEXSTREAM1FVATIPROC) load(userptr, "glVertexStream1fvATI");
    context->VertexStream1iATI = (PFNGLVERTEXSTREAM1IATIPROC) load(userptr, "glVertexStream1iATI");
    context->VertexStream1ivATI = (PFNGLVERTEXSTREAM1IVATIPROC) load(userptr, "glVertexStream1ivATI");
    context->VertexStream1sATI = (PFNGLVERTEXSTREAM1SATIPROC) load(userptr, "glVertexStream1sATI");
    context->VertexStream1svATI = (PFNGLVERTEXSTREAM1SVATIPROC) load(userptr, "glVertexStream1svATI");
    context->VertexStream2dATI = (PFNGLVERTEXSTREAM2DATIPROC) load(userptr, "glVertexStream2dATI");
    context->VertexStream2dvATI = (PFNGLVERTEXSTREAM2DVATIPROC) load(userptr, "glVertexStream2dvATI");
    context->VertexStream2fATI = (PFNGLVERTEXSTREAM2FATIPROC) load(userptr, "glVertexStream2fATI");
    context->VertexStream2fvATI = (PFNGLVERTEXSTREAM2FVATIPROC) load(userptr, "glVertexStream2fvATI");
    context->VertexStream2iATI = (PFNGLVERTEXSTREAM2IATIPROC) load(userptr, "glVertexStream2iATI");
    context->VertexStream2ivATI = (PFNGLVERTEXSTREAM2IVATIPROC) load(userptr, "glVertexStream2ivATI");
    context->VertexStream2sATI = (PFNGLVERTEXSTREAM2SATIPROC) load(userptr, "glVertexStream2sATI");
    context->VertexStream2svATI = (PFNGLVERTEXSTREAM2SVATIPROC) load(userptr, "glVertexStream2svATI");
    context->VertexStream3dATI = (PFNGLVERTEXSTREAM3DATIPROC) load(userptr, "glVertexStream3dATI");
    context->VertexStream3dvATI = (PFNGLVERTEXSTREAM3DVATIPROC) load(userptr, "glVertexStream3dvATI");
    context->VertexStream3fATI = (PFNGLVERTEXSTREAM3FATIPROC) load(userptr, "glVertexStream3fATI");
    context->VertexStream3fvATI = (PFNGLVERTEXSTREAM3FVATIPROC) load(userptr, "glVertexStream3fvATI");
    context->VertexStream3iATI = (PFNGLVERTEXSTREAM3IATIPROC) load(userptr, "glVertexStream3iATI");
    context->VertexStream3ivATI = (PFNGLVERTEXSTREAM3IVATIPROC) load(userptr, "glVertexStream3ivATI");
    context->VertexStream3sATI = (PFNGLVERTEXSTREAM3SATIPROC) load(userptr, "glVertexStream3sATI");
    context->VertexStream3svATI = (PFNGLVERTEXSTREAM3SVATIPROC) load(userptr, "glVertexStream3svATI");
    context->VertexStream4dATI = (PFNGLVERTEXSTREAM4DATIPROC) load(userptr, "glVertexStream4dATI");
    context->VertexStream4dvATI = (PFNGLVERTEXSTREAM4DVATIPROC) load(userptr, "glVertexStream4dvATI");
    context->VertexStream4fATI = (PFNGLVERTEXSTREAM4FATIPROC) load(userptr, "glVertexStream4fATI");
    context->VertexStream4fvATI = (PFNGLVERTEXSTREAM4FVATIPROC) load(userptr, "glVertexStream4fvATI");
    context->VertexStream4iATI = (PFNGLVERTEXSTREAM4IATIPROC) load(userptr, "glVertexStream4iATI");
    context->VertexStream4ivATI = (PFNGLVERTEXSTREAM4IVATIPROC) load(userptr, "glVertexStream4ivATI");
    context->VertexStream4sATI = (PFNGLVERTEXSTREAM4SATIPROC) load(userptr, "glVertexStream4sATI");
    context->VertexStream4svATI = (PFNGLVERTEXSTREAM4SVATIPROC) load(userptr, "glVertexStream4svATI");
}
static void glad_gl_load_GL_EXT_EGL_image_storage(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_EGL_image_storage) return;
    context->EGLImageTargetTexStorageEXT = (PFNGLEGLIMAGETARGETTEXSTORAGEEXTPROC) load(userptr, "glEGLImageTargetTexStorageEXT");
    context->EGLImageTargetTextureStorageEXT = (PFNGLEGLIMAGETARGETTEXTURESTORAGEEXTPROC) load(userptr, "glEGLImageTargetTextureStorageEXT");
}
static void glad_gl_load_GL_EXT_bindable_uniform(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_bindable_uniform) return;
    context->GetUniformBufferSizeEXT = (PFNGLGETUNIFORMBUFFERSIZEEXTPROC) load(userptr, "glGetUniformBufferSizeEXT");
    context->GetUniformOffsetEXT = (PFNGLGETUNIFORMOFFSETEXTPROC) load(userptr, "glGetUniformOffsetEXT");
    context->UniformBufferEXT = (PFNGLUNIFORMBUFFEREXTPROC) load(userptr, "glUniformBufferEXT");
}
static void glad_gl_load_GL_EXT_blend_color(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_blend_color) return;
    context->BlendColorEXT = (PFNGLBLENDCOLOREXTPROC) load(userptr, "glBlendColorEXT");
}
static void glad_gl_load_GL_EXT_blend_equation_separate(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_blend_equation_separate) return;
    context->BlendEquationSeparateEXT = (PFNGLBLENDEQUATIONSEPARATEEXTPROC) load(userptr, "glBlendEquationSeparateEXT");
}
static void glad_gl_load_GL_EXT_blend_func_separate(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_blend_func_separate) return;
    context->BlendFuncSeparateEXT = (PFNGLBLENDFUNCSEPARATEEXTPROC) load(userptr, "glBlendFuncSeparateEXT");
}
static void glad_gl_load_GL_EXT_blend_minmax(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_blend_minmax) return;
    context->BlendEquationEXT = (PFNGLBLENDEQUATIONEXTPROC) load(userptr, "glBlendEquationEXT");
}
static void glad_gl_load_GL_EXT_color_subtable(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_color_subtable) return;
    context->ColorSubTableEXT = (PFNGLCOLORSUBTABLEEXTPROC) load(userptr, "glColorSubTableEXT");
    context->CopyColorSubTableEXT = (PFNGLCOPYCOLORSUBTABLEEXTPROC) load(userptr, "glCopyColorSubTableEXT");
}
static void glad_gl_load_GL_EXT_compiled_vertex_array(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_compiled_vertex_array) return;
    context->LockArraysEXT = (PFNGLLOCKARRAYSEXTPROC) load(userptr, "glLockArraysEXT");
    context->UnlockArraysEXT = (PFNGLUNLOCKARRAYSEXTPROC) load(userptr, "glUnlockArraysEXT");
}
static void glad_gl_load_GL_EXT_convolution(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_convolution) return;
    context->ConvolutionFilter1DEXT = (PFNGLCONVOLUTIONFILTER1DEXTPROC) load(userptr, "glConvolutionFilter1DEXT");
    context->ConvolutionFilter2DEXT = (PFNGLCONVOLUTIONFILTER2DEXTPROC) load(userptr, "glConvolutionFilter2DEXT");
    context->ConvolutionParameterfEXT = (PFNGLCONVOLUTIONPARAMETERFEXTPROC) load(userptr, "glConvolutionParameterfEXT");
    context->ConvolutionParameterfvEXT = (PFNGLCONVOLUTIONPARAMETERFVEXTPROC) load(userptr, "glConvolutionParameterfvEXT");
    context->ConvolutionParameteriEXT = (PFNGLCONVOLUTIONPARAMETERIEXTPROC) load(userptr, "glConvolutionParameteriEXT");
    context->ConvolutionParameterivEXT = (PFNGLCONVOLUTIONPARAMETERIVEXTPROC) load(userptr, "glConvolutionParameterivEXT");
    context->CopyConvolutionFilter1DEXT = (PFNGLCOPYCONVOLUTIONFILTER1DEXTPROC) load(userptr, "glCopyConvolutionFilter1DEXT");
    context->CopyConvolutionFilter2DEXT = (PFNGLCOPYCONVOLUTIONFILTER2DEXTPROC) load(userptr, "glCopyConvolutionFilter2DEXT");
    context->GetConvolutionFilterEXT = (PFNGLGETCONVOLUTIONFILTEREXTPROC) load(userptr, "glGetConvolutionFilterEXT");
    context->GetConvolutionParameterfvEXT = (PFNGLGETCONVOLUTIONPARAMETERFVEXTPROC) load(userptr, "glGetConvolutionParameterfvEXT");
    context->GetConvolutionParameterivEXT = (PFNGLGETCONVOLUTIONPARAMETERIVEXTPROC) load(userptr, "glGetConvolutionParameterivEXT");
    context->GetSeparableFilterEXT = (PFNGLGETSEPARABLEFILTEREXTPROC) load(userptr, "glGetSeparableFilterEXT");
    context->SeparableFilter2DEXT = (PFNGLSEPARABLEFILTER2DEXTPROC) load(userptr, "glSeparableFilter2DEXT");
}
static void glad_gl_load_GL_EXT_coordinate_frame(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_coordinate_frame) return;
    context->Binormal3bEXT = (PFNGLBINORMAL3BEXTPROC) load(userptr, "glBinormal3bEXT");
    context->Binormal3bvEXT = (PFNGLBINORMAL3BVEXTPROC) load(userptr, "glBinormal3bvEXT");
    context->Binormal3dEXT = (PFNGLBINORMAL3DEXTPROC) load(userptr, "glBinormal3dEXT");
    context->Binormal3dvEXT = (PFNGLBINORMAL3DVEXTPROC) load(userptr, "glBinormal3dvEXT");
    context->Binormal3fEXT = (PFNGLBINORMAL3FEXTPROC) load(userptr, "glBinormal3fEXT");
    context->Binormal3fvEXT = (PFNGLBINORMAL3FVEXTPROC) load(userptr, "glBinormal3fvEXT");
    context->Binormal3iEXT = (PFNGLBINORMAL3IEXTPROC) load(userptr, "glBinormal3iEXT");
    context->Binormal3ivEXT = (PFNGLBINORMAL3IVEXTPROC) load(userptr, "glBinormal3ivEXT");
    context->Binormal3sEXT = (PFNGLBINORMAL3SEXTPROC) load(userptr, "glBinormal3sEXT");
    context->Binormal3svEXT = (PFNGLBINORMAL3SVEXTPROC) load(userptr, "glBinormal3svEXT");
    context->BinormalPointerEXT = (PFNGLBINORMALPOINTEREXTPROC) load(userptr, "glBinormalPointerEXT");
    context->Tangent3bEXT = (PFNGLTANGENT3BEXTPROC) load(userptr, "glTangent3bEXT");
    context->Tangent3bvEXT = (PFNGLTANGENT3BVEXTPROC) load(userptr, "glTangent3bvEXT");
    context->Tangent3dEXT = (PFNGLTANGENT3DEXTPROC) load(userptr, "glTangent3dEXT");
    context->Tangent3dvEXT = (PFNGLTANGENT3DVEXTPROC) load(userptr, "glTangent3dvEXT");
    context->Tangent3fEXT = (PFNGLTANGENT3FEXTPROC) load(userptr, "glTangent3fEXT");
    context->Tangent3fvEXT = (PFNGLTANGENT3FVEXTPROC) load(userptr, "glTangent3fvEXT");
    context->Tangent3iEXT = (PFNGLTANGENT3IEXTPROC) load(userptr, "glTangent3iEXT");
    context->Tangent3ivEXT = (PFNGLTANGENT3IVEXTPROC) load(userptr, "glTangent3ivEXT");
    context->Tangent3sEXT = (PFNGLTANGENT3SEXTPROC) load(userptr, "glTangent3sEXT");
    context->Tangent3svEXT = (PFNGLTANGENT3SVEXTPROC) load(userptr, "glTangent3svEXT");
    context->TangentPointerEXT = (PFNGLTANGENTPOINTEREXTPROC) load(userptr, "glTangentPointerEXT");
}
static void glad_gl_load_GL_EXT_copy_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_copy_texture) return;
    context->CopyTexImage1DEXT = (PFNGLCOPYTEXIMAGE1DEXTPROC) load(userptr, "glCopyTexImage1DEXT");
    context->CopyTexImage2DEXT = (PFNGLCOPYTEXIMAGE2DEXTPROC) load(userptr, "glCopyTexImage2DEXT");
    context->CopyTexSubImage1DEXT = (PFNGLCOPYTEXSUBIMAGE1DEXTPROC) load(userptr, "glCopyTexSubImage1DEXT");
    context->CopyTexSubImage2DEXT = (PFNGLCOPYTEXSUBIMAGE2DEXTPROC) load(userptr, "glCopyTexSubImage2DEXT");
    context->CopyTexSubImage3DEXT = (PFNGLCOPYTEXSUBIMAGE3DEXTPROC) load(userptr, "glCopyTexSubImage3DEXT");
}
static void glad_gl_load_GL_EXT_cull_vertex(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_cull_vertex) return;
    context->CullParameterdvEXT = (PFNGLCULLPARAMETERDVEXTPROC) load(userptr, "glCullParameterdvEXT");
    context->CullParameterfvEXT = (PFNGLCULLPARAMETERFVEXTPROC) load(userptr, "glCullParameterfvEXT");
}
static void glad_gl_load_GL_EXT_debug_label(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_debug_label) return;
    context->GetObjectLabelEXT = (PFNGLGETOBJECTLABELEXTPROC) load(userptr, "glGetObjectLabelEXT");
    context->LabelObjectEXT = (PFNGLLABELOBJECTEXTPROC) load(userptr, "glLabelObjectEXT");
}
static void glad_gl_load_GL_EXT_debug_marker(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_debug_marker) return;
    context->InsertEventMarkerEXT = (PFNGLINSERTEVENTMARKEREXTPROC) load(userptr, "glInsertEventMarkerEXT");
    context->PopGroupMarkerEXT = (PFNGLPOPGROUPMARKEREXTPROC) load(userptr, "glPopGroupMarkerEXT");
    context->PushGroupMarkerEXT = (PFNGLPUSHGROUPMARKEREXTPROC) load(userptr, "glPushGroupMarkerEXT");
}
static void glad_gl_load_GL_EXT_depth_bounds_test(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_depth_bounds_test) return;
    context->DepthBoundsEXT = (PFNGLDEPTHBOUNDSEXTPROC) load(userptr, "glDepthBoundsEXT");
}
static void glad_gl_load_GL_EXT_direct_state_access(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_direct_state_access) return;
    context->BindMultiTextureEXT = (PFNGLBINDMULTITEXTUREEXTPROC) load(userptr, "glBindMultiTextureEXT");
    context->CheckNamedFramebufferStatusEXT = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC) load(userptr, "glCheckNamedFramebufferStatusEXT");
    context->ClearNamedBufferDataEXT = (PFNGLCLEARNAMEDBUFFERDATAEXTPROC) load(userptr, "glClearNamedBufferDataEXT");
    context->ClearNamedBufferSubDataEXT = (PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC) load(userptr, "glClearNamedBufferSubDataEXT");
    context->ClientAttribDefaultEXT = (PFNGLCLIENTATTRIBDEFAULTEXTPROC) load(userptr, "glClientAttribDefaultEXT");
    context->CompressedMultiTexImage1DEXT = (PFNGLCOMPRESSEDMULTITEXIMAGE1DEXTPROC) load(userptr, "glCompressedMultiTexImage1DEXT");
    context->CompressedMultiTexImage2DEXT = (PFNGLCOMPRESSEDMULTITEXIMAGE2DEXTPROC) load(userptr, "glCompressedMultiTexImage2DEXT");
    context->CompressedMultiTexImage3DEXT = (PFNGLCOMPRESSEDMULTITEXIMAGE3DEXTPROC) load(userptr, "glCompressedMultiTexImage3DEXT");
    context->CompressedMultiTexSubImage1DEXT = (PFNGLCOMPRESSEDMULTITEXSUBIMAGE1DEXTPROC) load(userptr, "glCompressedMultiTexSubImage1DEXT");
    context->CompressedMultiTexSubImage2DEXT = (PFNGLCOMPRESSEDMULTITEXSUBIMAGE2DEXTPROC) load(userptr, "glCompressedMultiTexSubImage2DEXT");
    context->CompressedMultiTexSubImage3DEXT = (PFNGLCOMPRESSEDMULTITEXSUBIMAGE3DEXTPROC) load(userptr, "glCompressedMultiTexSubImage3DEXT");
    context->CompressedTextureImage1DEXT = (PFNGLCOMPRESSEDTEXTUREIMAGE1DEXTPROC) load(userptr, "glCompressedTextureImage1DEXT");
    context->CompressedTextureImage2DEXT = (PFNGLCOMPRESSEDTEXTUREIMAGE2DEXTPROC) load(userptr, "glCompressedTextureImage2DEXT");
    context->CompressedTextureImage3DEXT = (PFNGLCOMPRESSEDTEXTUREIMAGE3DEXTPROC) load(userptr, "glCompressedTextureImage3DEXT");
    context->CompressedTextureSubImage1DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC) load(userptr, "glCompressedTextureSubImage1DEXT");
    context->CompressedTextureSubImage2DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC) load(userptr, "glCompressedTextureSubImage2DEXT");
    context->CompressedTextureSubImage3DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC) load(userptr, "glCompressedTextureSubImage3DEXT");
    context->CopyMultiTexImage1DEXT = (PFNGLCOPYMULTITEXIMAGE1DEXTPROC) load(userptr, "glCopyMultiTexImage1DEXT");
    context->CopyMultiTexImage2DEXT = (PFNGLCOPYMULTITEXIMAGE2DEXTPROC) load(userptr, "glCopyMultiTexImage2DEXT");
    context->CopyMultiTexSubImage1DEXT = (PFNGLCOPYMULTITEXSUBIMAGE1DEXTPROC) load(userptr, "glCopyMultiTexSubImage1DEXT");
    context->CopyMultiTexSubImage2DEXT = (PFNGLCOPYMULTITEXSUBIMAGE2DEXTPROC) load(userptr, "glCopyMultiTexSubImage2DEXT");
    context->CopyMultiTexSubImage3DEXT = (PFNGLCOPYMULTITEXSUBIMAGE3DEXTPROC) load(userptr, "glCopyMultiTexSubImage3DEXT");
    context->CopyTextureImage1DEXT = (PFNGLCOPYTEXTUREIMAGE1DEXTPROC) load(userptr, "glCopyTextureImage1DEXT");
    context->CopyTextureImage2DEXT = (PFNGLCOPYTEXTUREIMAGE2DEXTPROC) load(userptr, "glCopyTextureImage2DEXT");
    context->CopyTextureSubImage1DEXT = (PFNGLCOPYTEXTURESUBIMAGE1DEXTPROC) load(userptr, "glCopyTextureSubImage1DEXT");
    context->CopyTextureSubImage2DEXT = (PFNGLCOPYTEXTURESUBIMAGE2DEXTPROC) load(userptr, "glCopyTextureSubImage2DEXT");
    context->CopyTextureSubImage3DEXT = (PFNGLCOPYTEXTURESUBIMAGE3DEXTPROC) load(userptr, "glCopyTextureSubImage3DEXT");
    context->DisableClientStateIndexedEXT = (PFNGLDISABLECLIENTSTATEINDEXEDEXTPROC) load(userptr, "glDisableClientStateIndexedEXT");
    context->DisableClientStateiEXT = (PFNGLDISABLECLIENTSTATEIEXTPROC) load(userptr, "glDisableClientStateiEXT");
    context->DisableIndexedEXT = (PFNGLDISABLEINDEXEDEXTPROC) load(userptr, "glDisableIndexedEXT");
    context->DisableVertexArrayAttribEXT = (PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC) load(userptr, "glDisableVertexArrayAttribEXT");
    context->DisableVertexArrayEXT = (PFNGLDISABLEVERTEXARRAYEXTPROC) load(userptr, "glDisableVertexArrayEXT");
    context->EnableClientStateIndexedEXT = (PFNGLENABLECLIENTSTATEINDEXEDEXTPROC) load(userptr, "glEnableClientStateIndexedEXT");
    context->EnableClientStateiEXT = (PFNGLENABLECLIENTSTATEIEXTPROC) load(userptr, "glEnableClientStateiEXT");
    context->EnableIndexedEXT = (PFNGLENABLEINDEXEDEXTPROC) load(userptr, "glEnableIndexedEXT");
    context->EnableVertexArrayAttribEXT = (PFNGLENABLEVERTEXARRAYATTRIBEXTPROC) load(userptr, "glEnableVertexArrayAttribEXT");
    context->EnableVertexArrayEXT = (PFNGLENABLEVERTEXARRAYEXTPROC) load(userptr, "glEnableVertexArrayEXT");
    context->FlushMappedNamedBufferRangeEXT = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC) load(userptr, "glFlushMappedNamedBufferRangeEXT");
    context->FramebufferDrawBufferEXT = (PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC) load(userptr, "glFramebufferDrawBufferEXT");
    context->FramebufferDrawBuffersEXT = (PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC) load(userptr, "glFramebufferDrawBuffersEXT");
    context->FramebufferReadBufferEXT = (PFNGLFRAMEBUFFERREADBUFFEREXTPROC) load(userptr, "glFramebufferReadBufferEXT");
    context->GenerateMultiTexMipmapEXT = (PFNGLGENERATEMULTITEXMIPMAPEXTPROC) load(userptr, "glGenerateMultiTexMipmapEXT");
    context->GenerateTextureMipmapEXT = (PFNGLGENERATETEXTUREMIPMAPEXTPROC) load(userptr, "glGenerateTextureMipmapEXT");
    context->GetBooleanIndexedvEXT = (PFNGLGETBOOLEANINDEXEDVEXTPROC) load(userptr, "glGetBooleanIndexedvEXT");
    context->GetCompressedMultiTexImageEXT = (PFNGLGETCOMPRESSEDMULTITEXIMAGEEXTPROC) load(userptr, "glGetCompressedMultiTexImageEXT");
    context->GetCompressedTextureImageEXT = (PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC) load(userptr, "glGetCompressedTextureImageEXT");
    context->GetDoubleIndexedvEXT = (PFNGLGETDOUBLEINDEXEDVEXTPROC) load(userptr, "glGetDoubleIndexedvEXT");
    context->GetDoublei_vEXT = (PFNGLGETDOUBLEI_VEXTPROC) load(userptr, "glGetDoublei_vEXT");
    context->GetFloatIndexedvEXT = (PFNGLGETFLOATINDEXEDVEXTPROC) load(userptr, "glGetFloatIndexedvEXT");
    context->GetFloati_vEXT = (PFNGLGETFLOATI_VEXTPROC) load(userptr, "glGetFloati_vEXT");
    context->GetFramebufferParameterivEXT = (PFNGLGETFRAMEBUFFERPARAMETERIVEXTPROC) load(userptr, "glGetFramebufferParameterivEXT");
    context->GetIntegerIndexedvEXT = (PFNGLGETINTEGERINDEXEDVEXTPROC) load(userptr, "glGetIntegerIndexedvEXT");
    context->GetMultiTexEnvfvEXT = (PFNGLGETMULTITEXENVFVEXTPROC) load(userptr, "glGetMultiTexEnvfvEXT");
    context->GetMultiTexEnvivEXT = (PFNGLGETMULTITEXENVIVEXTPROC) load(userptr, "glGetMultiTexEnvivEXT");
    context->GetMultiTexGendvEXT = (PFNGLGETMULTITEXGENDVEXTPROC) load(userptr, "glGetMultiTexGendvEXT");
    context->GetMultiTexGenfvEXT = (PFNGLGETMULTITEXGENFVEXTPROC) load(userptr, "glGetMultiTexGenfvEXT");
    context->GetMultiTexGenivEXT = (PFNGLGETMULTITEXGENIVEXTPROC) load(userptr, "glGetMultiTexGenivEXT");
    context->GetMultiTexImageEXT = (PFNGLGETMULTITEXIMAGEEXTPROC) load(userptr, "glGetMultiTexImageEXT");
    context->GetMultiTexLevelParameterfvEXT = (PFNGLGETMULTITEXLEVELPARAMETERFVEXTPROC) load(userptr, "glGetMultiTexLevelParameterfvEXT");
    context->GetMultiTexLevelParameterivEXT = (PFNGLGETMULTITEXLEVELPARAMETERIVEXTPROC) load(userptr, "glGetMultiTexLevelParameterivEXT");
    context->GetMultiTexParameterIivEXT = (PFNGLGETMULTITEXPARAMETERIIVEXTPROC) load(userptr, "glGetMultiTexParameterIivEXT");
    context->GetMultiTexParameterIuivEXT = (PFNGLGETMULTITEXPARAMETERIUIVEXTPROC) load(userptr, "glGetMultiTexParameterIuivEXT");
    context->GetMultiTexParameterfvEXT = (PFNGLGETMULTITEXPARAMETERFVEXTPROC) load(userptr, "glGetMultiTexParameterfvEXT");
    context->GetMultiTexParameterivEXT = (PFNGLGETMULTITEXPARAMETERIVEXTPROC) load(userptr, "glGetMultiTexParameterivEXT");
    context->GetNamedBufferParameterivEXT = (PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC) load(userptr, "glGetNamedBufferParameterivEXT");
    context->GetNamedBufferPointervEXT = (PFNGLGETNAMEDBUFFERPOINTERVEXTPROC) load(userptr, "glGetNamedBufferPointervEXT");
    context->GetNamedBufferSubDataEXT = (PFNGLGETNAMEDBUFFERSUBDATAEXTPROC) load(userptr, "glGetNamedBufferSubDataEXT");
    context->GetNamedFramebufferAttachmentParameterivEXT = (PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC) load(userptr, "glGetNamedFramebufferAttachmentParameterivEXT");
    context->GetNamedFramebufferParameterivEXT = (PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVEXTPROC) load(userptr, "glGetNamedFramebufferParameterivEXT");
    context->GetNamedProgramLocalParameterIivEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERIIVEXTPROC) load(userptr, "glGetNamedProgramLocalParameterIivEXT");
    context->GetNamedProgramLocalParameterIuivEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERIUIVEXTPROC) load(userptr, "glGetNamedProgramLocalParameterIuivEXT");
    context->GetNamedProgramLocalParameterdvEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERDVEXTPROC) load(userptr, "glGetNamedProgramLocalParameterdvEXT");
    context->GetNamedProgramLocalParameterfvEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERFVEXTPROC) load(userptr, "glGetNamedProgramLocalParameterfvEXT");
    context->GetNamedProgramStringEXT = (PFNGLGETNAMEDPROGRAMSTRINGEXTPROC) load(userptr, "glGetNamedProgramStringEXT");
    context->GetNamedProgramivEXT = (PFNGLGETNAMEDPROGRAMIVEXTPROC) load(userptr, "glGetNamedProgramivEXT");
    context->GetNamedRenderbufferParameterivEXT = (PFNGLGETNAMEDRENDERBUFFERPARAMETERIVEXTPROC) load(userptr, "glGetNamedRenderbufferParameterivEXT");
    context->GetPointerIndexedvEXT = (PFNGLGETPOINTERINDEXEDVEXTPROC) load(userptr, "glGetPointerIndexedvEXT");
    context->GetPointeri_vEXT = (PFNGLGETPOINTERI_VEXTPROC) load(userptr, "glGetPointeri_vEXT");
    context->GetTextureImageEXT = (PFNGLGETTEXTUREIMAGEEXTPROC) load(userptr, "glGetTextureImageEXT");
    context->GetTextureLevelParameterfvEXT = (PFNGLGETTEXTURELEVELPARAMETERFVEXTPROC) load(userptr, "glGetTextureLevelParameterfvEXT");
    context->GetTextureLevelParameterivEXT = (PFNGLGETTEXTURELEVELPARAMETERIVEXTPROC) load(userptr, "glGetTextureLevelParameterivEXT");
    context->GetTextureParameterIivEXT = (PFNGLGETTEXTUREPARAMETERIIVEXTPROC) load(userptr, "glGetTextureParameterIivEXT");
    context->GetTextureParameterIuivEXT = (PFNGLGETTEXTUREPARAMETERIUIVEXTPROC) load(userptr, "glGetTextureParameterIuivEXT");
    context->GetTextureParameterfvEXT = (PFNGLGETTEXTUREPARAMETERFVEXTPROC) load(userptr, "glGetTextureParameterfvEXT");
    context->GetTextureParameterivEXT = (PFNGLGETTEXTUREPARAMETERIVEXTPROC) load(userptr, "glGetTextureParameterivEXT");
    context->GetVertexArrayIntegeri_vEXT = (PFNGLGETVERTEXARRAYINTEGERI_VEXTPROC) load(userptr, "glGetVertexArrayIntegeri_vEXT");
    context->GetVertexArrayIntegervEXT = (PFNGLGETVERTEXARRAYINTEGERVEXTPROC) load(userptr, "glGetVertexArrayIntegervEXT");
    context->GetVertexArrayPointeri_vEXT = (PFNGLGETVERTEXARRAYPOINTERI_VEXTPROC) load(userptr, "glGetVertexArrayPointeri_vEXT");
    context->GetVertexArrayPointervEXT = (PFNGLGETVERTEXARRAYPOINTERVEXTPROC) load(userptr, "glGetVertexArrayPointervEXT");
    context->IsEnabledIndexedEXT = (PFNGLISENABLEDINDEXEDEXTPROC) load(userptr, "glIsEnabledIndexedEXT");
    context->MapNamedBufferEXT = (PFNGLMAPNAMEDBUFFEREXTPROC) load(userptr, "glMapNamedBufferEXT");
    context->MapNamedBufferRangeEXT = (PFNGLMAPNAMEDBUFFERRANGEEXTPROC) load(userptr, "glMapNamedBufferRangeEXT");
    context->MatrixFrustumEXT = (PFNGLMATRIXFRUSTUMEXTPROC) load(userptr, "glMatrixFrustumEXT");
    context->MatrixLoadIdentityEXT = (PFNGLMATRIXLOADIDENTITYEXTPROC) load(userptr, "glMatrixLoadIdentityEXT");
    context->MatrixLoadTransposedEXT = (PFNGLMATRIXLOADTRANSPOSEDEXTPROC) load(userptr, "glMatrixLoadTransposedEXT");
    context->MatrixLoadTransposefEXT = (PFNGLMATRIXLOADTRANSPOSEFEXTPROC) load(userptr, "glMatrixLoadTransposefEXT");
    context->MatrixLoaddEXT = (PFNGLMATRIXLOADDEXTPROC) load(userptr, "glMatrixLoaddEXT");
    context->MatrixLoadfEXT = (PFNGLMATRIXLOADFEXTPROC) load(userptr, "glMatrixLoadfEXT");
    context->MatrixMultTransposedEXT = (PFNGLMATRIXMULTTRANSPOSEDEXTPROC) load(userptr, "glMatrixMultTransposedEXT");
    context->MatrixMultTransposefEXT = (PFNGLMATRIXMULTTRANSPOSEFEXTPROC) load(userptr, "glMatrixMultTransposefEXT");
    context->MatrixMultdEXT = (PFNGLMATRIXMULTDEXTPROC) load(userptr, "glMatrixMultdEXT");
    context->MatrixMultfEXT = (PFNGLMATRIXMULTFEXTPROC) load(userptr, "glMatrixMultfEXT");
    context->MatrixOrthoEXT = (PFNGLMATRIXORTHOEXTPROC) load(userptr, "glMatrixOrthoEXT");
    context->MatrixPopEXT = (PFNGLMATRIXPOPEXTPROC) load(userptr, "glMatrixPopEXT");
    context->MatrixPushEXT = (PFNGLMATRIXPUSHEXTPROC) load(userptr, "glMatrixPushEXT");
    context->MatrixRotatedEXT = (PFNGLMATRIXROTATEDEXTPROC) load(userptr, "glMatrixRotatedEXT");
    context->MatrixRotatefEXT = (PFNGLMATRIXROTATEFEXTPROC) load(userptr, "glMatrixRotatefEXT");
    context->MatrixScaledEXT = (PFNGLMATRIXSCALEDEXTPROC) load(userptr, "glMatrixScaledEXT");
    context->MatrixScalefEXT = (PFNGLMATRIXSCALEFEXTPROC) load(userptr, "glMatrixScalefEXT");
    context->MatrixTranslatedEXT = (PFNGLMATRIXTRANSLATEDEXTPROC) load(userptr, "glMatrixTranslatedEXT");
    context->MatrixTranslatefEXT = (PFNGLMATRIXTRANSLATEFEXTPROC) load(userptr, "glMatrixTranslatefEXT");
    context->MultiTexBufferEXT = (PFNGLMULTITEXBUFFEREXTPROC) load(userptr, "glMultiTexBufferEXT");
    context->MultiTexCoordPointerEXT = (PFNGLMULTITEXCOORDPOINTEREXTPROC) load(userptr, "glMultiTexCoordPointerEXT");
    context->MultiTexEnvfEXT = (PFNGLMULTITEXENVFEXTPROC) load(userptr, "glMultiTexEnvfEXT");
    context->MultiTexEnvfvEXT = (PFNGLMULTITEXENVFVEXTPROC) load(userptr, "glMultiTexEnvfvEXT");
    context->MultiTexEnviEXT = (PFNGLMULTITEXENVIEXTPROC) load(userptr, "glMultiTexEnviEXT");
    context->MultiTexEnvivEXT = (PFNGLMULTITEXENVIVEXTPROC) load(userptr, "glMultiTexEnvivEXT");
    context->MultiTexGendEXT = (PFNGLMULTITEXGENDEXTPROC) load(userptr, "glMultiTexGendEXT");
    context->MultiTexGendvEXT = (PFNGLMULTITEXGENDVEXTPROC) load(userptr, "glMultiTexGendvEXT");
    context->MultiTexGenfEXT = (PFNGLMULTITEXGENFEXTPROC) load(userptr, "glMultiTexGenfEXT");
    context->MultiTexGenfvEXT = (PFNGLMULTITEXGENFVEXTPROC) load(userptr, "glMultiTexGenfvEXT");
    context->MultiTexGeniEXT = (PFNGLMULTITEXGENIEXTPROC) load(userptr, "glMultiTexGeniEXT");
    context->MultiTexGenivEXT = (PFNGLMULTITEXGENIVEXTPROC) load(userptr, "glMultiTexGenivEXT");
    context->MultiTexImage1DEXT = (PFNGLMULTITEXIMAGE1DEXTPROC) load(userptr, "glMultiTexImage1DEXT");
    context->MultiTexImage2DEXT = (PFNGLMULTITEXIMAGE2DEXTPROC) load(userptr, "glMultiTexImage2DEXT");
    context->MultiTexImage3DEXT = (PFNGLMULTITEXIMAGE3DEXTPROC) load(userptr, "glMultiTexImage3DEXT");
    context->MultiTexParameterIivEXT = (PFNGLMULTITEXPARAMETERIIVEXTPROC) load(userptr, "glMultiTexParameterIivEXT");
    context->MultiTexParameterIuivEXT = (PFNGLMULTITEXPARAMETERIUIVEXTPROC) load(userptr, "glMultiTexParameterIuivEXT");
    context->MultiTexParameterfEXT = (PFNGLMULTITEXPARAMETERFEXTPROC) load(userptr, "glMultiTexParameterfEXT");
    context->MultiTexParameterfvEXT = (PFNGLMULTITEXPARAMETERFVEXTPROC) load(userptr, "glMultiTexParameterfvEXT");
    context->MultiTexParameteriEXT = (PFNGLMULTITEXPARAMETERIEXTPROC) load(userptr, "glMultiTexParameteriEXT");
    context->MultiTexParameterivEXT = (PFNGLMULTITEXPARAMETERIVEXTPROC) load(userptr, "glMultiTexParameterivEXT");
    context->MultiTexRenderbufferEXT = (PFNGLMULTITEXRENDERBUFFEREXTPROC) load(userptr, "glMultiTexRenderbufferEXT");
    context->MultiTexSubImage1DEXT = (PFNGLMULTITEXSUBIMAGE1DEXTPROC) load(userptr, "glMultiTexSubImage1DEXT");
    context->MultiTexSubImage2DEXT = (PFNGLMULTITEXSUBIMAGE2DEXTPROC) load(userptr, "glMultiTexSubImage2DEXT");
    context->MultiTexSubImage3DEXT = (PFNGLMULTITEXSUBIMAGE3DEXTPROC) load(userptr, "glMultiTexSubImage3DEXT");
    context->NamedBufferDataEXT = (PFNGLNAMEDBUFFERDATAEXTPROC) load(userptr, "glNamedBufferDataEXT");
    context->NamedBufferStorageEXT = (PFNGLNAMEDBUFFERSTORAGEEXTPROC) load(userptr, "glNamedBufferStorageEXT");
    context->NamedBufferSubDataEXT = (PFNGLNAMEDBUFFERSUBDATAEXTPROC) load(userptr, "glNamedBufferSubDataEXT");
    context->NamedCopyBufferSubDataEXT = (PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC) load(userptr, "glNamedCopyBufferSubDataEXT");
    context->NamedFramebufferParameteriEXT = (PFNGLNAMEDFRAMEBUFFERPARAMETERIEXTPROC) load(userptr, "glNamedFramebufferParameteriEXT");
    context->NamedFramebufferRenderbufferEXT = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC) load(userptr, "glNamedFramebufferRenderbufferEXT");
    context->NamedFramebufferTexture1DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE1DEXTPROC) load(userptr, "glNamedFramebufferTexture1DEXT");
    context->NamedFramebufferTexture2DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC) load(userptr, "glNamedFramebufferTexture2DEXT");
    context->NamedFramebufferTexture3DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE3DEXTPROC) load(userptr, "glNamedFramebufferTexture3DEXT");
    context->NamedFramebufferTextureEXT = (PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC) load(userptr, "glNamedFramebufferTextureEXT");
    context->NamedFramebufferTextureFaceEXT = (PFNGLNAMEDFRAMEBUFFERTEXTUREFACEEXTPROC) load(userptr, "glNamedFramebufferTextureFaceEXT");
    context->NamedFramebufferTextureLayerEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC) load(userptr, "glNamedFramebufferTextureLayerEXT");
    context->NamedProgramLocalParameter4dEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4DEXTPROC) load(userptr, "glNamedProgramLocalParameter4dEXT");
    context->NamedProgramLocalParameter4dvEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4DVEXTPROC) load(userptr, "glNamedProgramLocalParameter4dvEXT");
    context->NamedProgramLocalParameter4fEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4FEXTPROC) load(userptr, "glNamedProgramLocalParameter4fEXT");
    context->NamedProgramLocalParameter4fvEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4FVEXTPROC) load(userptr, "glNamedProgramLocalParameter4fvEXT");
    context->NamedProgramLocalParameterI4iEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4IEXTPROC) load(userptr, "glNamedProgramLocalParameterI4iEXT");
    context->NamedProgramLocalParameterI4ivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4IVEXTPROC) load(userptr, "glNamedProgramLocalParameterI4ivEXT");
    context->NamedProgramLocalParameterI4uiEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIEXTPROC) load(userptr, "glNamedProgramLocalParameterI4uiEXT");
    context->NamedProgramLocalParameterI4uivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIVEXTPROC) load(userptr, "glNamedProgramLocalParameterI4uivEXT");
    context->NamedProgramLocalParameters4fvEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERS4FVEXTPROC) load(userptr, "glNamedProgramLocalParameters4fvEXT");
    context->NamedProgramLocalParametersI4ivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERSI4IVEXTPROC) load(userptr, "glNamedProgramLocalParametersI4ivEXT");
    context->NamedProgramLocalParametersI4uivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERSI4UIVEXTPROC) load(userptr, "glNamedProgramLocalParametersI4uivEXT");
    context->NamedProgramStringEXT = (PFNGLNAMEDPROGRAMSTRINGEXTPROC) load(userptr, "glNamedProgramStringEXT");
    context->NamedRenderbufferStorageEXT = (PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC) load(userptr, "glNamedRenderbufferStorageEXT");
    context->NamedRenderbufferStorageMultisampleCoverageEXT = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLECOVERAGEEXTPROC) load(userptr, "glNamedRenderbufferStorageMultisampleCoverageEXT");
    context->NamedRenderbufferStorageMultisampleEXT = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC) load(userptr, "glNamedRenderbufferStorageMultisampleEXT");
    context->ProgramUniform1dEXT = (PFNGLPROGRAMUNIFORM1DEXTPROC) load(userptr, "glProgramUniform1dEXT");
    context->ProgramUniform1dvEXT = (PFNGLPROGRAMUNIFORM1DVEXTPROC) load(userptr, "glProgramUniform1dvEXT");
    context->ProgramUniform1fEXT = (PFNGLPROGRAMUNIFORM1FEXTPROC) load(userptr, "glProgramUniform1fEXT");
    context->ProgramUniform1fvEXT = (PFNGLPROGRAMUNIFORM1FVEXTPROC) load(userptr, "glProgramUniform1fvEXT");
    context->ProgramUniform1iEXT = (PFNGLPROGRAMUNIFORM1IEXTPROC) load(userptr, "glProgramUniform1iEXT");
    context->ProgramUniform1ivEXT = (PFNGLPROGRAMUNIFORM1IVEXTPROC) load(userptr, "glProgramUniform1ivEXT");
    context->ProgramUniform1uiEXT = (PFNGLPROGRAMUNIFORM1UIEXTPROC) load(userptr, "glProgramUniform1uiEXT");
    context->ProgramUniform1uivEXT = (PFNGLPROGRAMUNIFORM1UIVEXTPROC) load(userptr, "glProgramUniform1uivEXT");
    context->ProgramUniform2dEXT = (PFNGLPROGRAMUNIFORM2DEXTPROC) load(userptr, "glProgramUniform2dEXT");
    context->ProgramUniform2dvEXT = (PFNGLPROGRAMUNIFORM2DVEXTPROC) load(userptr, "glProgramUniform2dvEXT");
    context->ProgramUniform2fEXT = (PFNGLPROGRAMUNIFORM2FEXTPROC) load(userptr, "glProgramUniform2fEXT");
    context->ProgramUniform2fvEXT = (PFNGLPROGRAMUNIFORM2FVEXTPROC) load(userptr, "glProgramUniform2fvEXT");
    context->ProgramUniform2iEXT = (PFNGLPROGRAMUNIFORM2IEXTPROC) load(userptr, "glProgramUniform2iEXT");
    context->ProgramUniform2ivEXT = (PFNGLPROGRAMUNIFORM2IVEXTPROC) load(userptr, "glProgramUniform2ivEXT");
    context->ProgramUniform2uiEXT = (PFNGLPROGRAMUNIFORM2UIEXTPROC) load(userptr, "glProgramUniform2uiEXT");
    context->ProgramUniform2uivEXT = (PFNGLPROGRAMUNIFORM2UIVEXTPROC) load(userptr, "glProgramUniform2uivEXT");
    context->ProgramUniform3dEXT = (PFNGLPROGRAMUNIFORM3DEXTPROC) load(userptr, "glProgramUniform3dEXT");
    context->ProgramUniform3dvEXT = (PFNGLPROGRAMUNIFORM3DVEXTPROC) load(userptr, "glProgramUniform3dvEXT");
    context->ProgramUniform3fEXT = (PFNGLPROGRAMUNIFORM3FEXTPROC) load(userptr, "glProgramUniform3fEXT");
    context->ProgramUniform3fvEXT = (PFNGLPROGRAMUNIFORM3FVEXTPROC) load(userptr, "glProgramUniform3fvEXT");
    context->ProgramUniform3iEXT = (PFNGLPROGRAMUNIFORM3IEXTPROC) load(userptr, "glProgramUniform3iEXT");
    context->ProgramUniform3ivEXT = (PFNGLPROGRAMUNIFORM3IVEXTPROC) load(userptr, "glProgramUniform3ivEXT");
    context->ProgramUniform3uiEXT = (PFNGLPROGRAMUNIFORM3UIEXTPROC) load(userptr, "glProgramUniform3uiEXT");
    context->ProgramUniform3uivEXT = (PFNGLPROGRAMUNIFORM3UIVEXTPROC) load(userptr, "glProgramUniform3uivEXT");
    context->ProgramUniform4dEXT = (PFNGLPROGRAMUNIFORM4DEXTPROC) load(userptr, "glProgramUniform4dEXT");
    context->ProgramUniform4dvEXT = (PFNGLPROGRAMUNIFORM4DVEXTPROC) load(userptr, "glProgramUniform4dvEXT");
    context->ProgramUniform4fEXT = (PFNGLPROGRAMUNIFORM4FEXTPROC) load(userptr, "glProgramUniform4fEXT");
    context->ProgramUniform4fvEXT = (PFNGLPROGRAMUNIFORM4FVEXTPROC) load(userptr, "glProgramUniform4fvEXT");
    context->ProgramUniform4iEXT = (PFNGLPROGRAMUNIFORM4IEXTPROC) load(userptr, "glProgramUniform4iEXT");
    context->ProgramUniform4ivEXT = (PFNGLPROGRAMUNIFORM4IVEXTPROC) load(userptr, "glProgramUniform4ivEXT");
    context->ProgramUniform4uiEXT = (PFNGLPROGRAMUNIFORM4UIEXTPROC) load(userptr, "glProgramUniform4uiEXT");
    context->ProgramUniform4uivEXT = (PFNGLPROGRAMUNIFORM4UIVEXTPROC) load(userptr, "glProgramUniform4uivEXT");
    context->ProgramUniformMatrix2dvEXT = (PFNGLPROGRAMUNIFORMMATRIX2DVEXTPROC) load(userptr, "glProgramUniformMatrix2dvEXT");
    context->ProgramUniformMatrix2fvEXT = (PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC) load(userptr, "glProgramUniformMatrix2fvEXT");
    context->ProgramUniformMatrix2x3dvEXT = (PFNGLPROGRAMUNIFORMMATRIX2X3DVEXTPROC) load(userptr, "glProgramUniformMatrix2x3dvEXT");
    context->ProgramUniformMatrix2x3fvEXT = (PFNGLPROGRAMUNIFORMMATRIX2X3FVEXTPROC) load(userptr, "glProgramUniformMatrix2x3fvEXT");
    context->ProgramUniformMatrix2x4dvEXT = (PFNGLPROGRAMUNIFORMMATRIX2X4DVEXTPROC) load(userptr, "glProgramUniformMatrix2x4dvEXT");
    context->ProgramUniformMatrix2x4fvEXT = (PFNGLPROGRAMUNIFORMMATRIX2X4FVEXTPROC) load(userptr, "glProgramUniformMatrix2x4fvEXT");
    context->ProgramUniformMatrix3dvEXT = (PFNGLPROGRAMUNIFORMMATRIX3DVEXTPROC) load(userptr, "glProgramUniformMatrix3dvEXT");
    context->ProgramUniformMatrix3fvEXT = (PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC) load(userptr, "glProgramUniformMatrix3fvEXT");
    context->ProgramUniformMatrix3x2dvEXT = (PFNGLPROGRAMUNIFORMMATRIX3X2DVEXTPROC) load(userptr, "glProgramUniformMatrix3x2dvEXT");
    context->ProgramUniformMatrix3x2fvEXT = (PFNGLPROGRAMUNIFORMMATRIX3X2FVEXTPROC) load(userptr, "glProgramUniformMatrix3x2fvEXT");
    context->ProgramUniformMatrix3x4dvEXT = (PFNGLPROGRAMUNIFORMMATRIX3X4DVEXTPROC) load(userptr, "glProgramUniformMatrix3x4dvEXT");
    context->ProgramUniformMatrix3x4fvEXT = (PFNGLPROGRAMUNIFORMMATRIX3X4FVEXTPROC) load(userptr, "glProgramUniformMatrix3x4fvEXT");
    context->ProgramUniformMatrix4dvEXT = (PFNGLPROGRAMUNIFORMMATRIX4DVEXTPROC) load(userptr, "glProgramUniformMatrix4dvEXT");
    context->ProgramUniformMatrix4fvEXT = (PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC) load(userptr, "glProgramUniformMatrix4fvEXT");
    context->ProgramUniformMatrix4x2dvEXT = (PFNGLPROGRAMUNIFORMMATRIX4X2DVEXTPROC) load(userptr, "glProgramUniformMatrix4x2dvEXT");
    context->ProgramUniformMatrix4x2fvEXT = (PFNGLPROGRAMUNIFORMMATRIX4X2FVEXTPROC) load(userptr, "glProgramUniformMatrix4x2fvEXT");
    context->ProgramUniformMatrix4x3dvEXT = (PFNGLPROGRAMUNIFORMMATRIX4X3DVEXTPROC) load(userptr, "glProgramUniformMatrix4x3dvEXT");
    context->ProgramUniformMatrix4x3fvEXT = (PFNGLPROGRAMUNIFORMMATRIX4X3FVEXTPROC) load(userptr, "glProgramUniformMatrix4x3fvEXT");
    context->PushClientAttribDefaultEXT = (PFNGLPUSHCLIENTATTRIBDEFAULTEXTPROC) load(userptr, "glPushClientAttribDefaultEXT");
    context->TextureBufferEXT = (PFNGLTEXTUREBUFFEREXTPROC) load(userptr, "glTextureBufferEXT");
    context->TextureBufferRangeEXT = (PFNGLTEXTUREBUFFERRANGEEXTPROC) load(userptr, "glTextureBufferRangeEXT");
    context->TextureImage1DEXT = (PFNGLTEXTUREIMAGE1DEXTPROC) load(userptr, "glTextureImage1DEXT");
    context->TextureImage2DEXT = (PFNGLTEXTUREIMAGE2DEXTPROC) load(userptr, "glTextureImage2DEXT");
    context->TextureImage3DEXT = (PFNGLTEXTUREIMAGE3DEXTPROC) load(userptr, "glTextureImage3DEXT");
    context->TexturePageCommitmentEXT = (PFNGLTEXTUREPAGECOMMITMENTEXTPROC) load(userptr, "glTexturePageCommitmentEXT");
    context->TextureParameterIivEXT = (PFNGLTEXTUREPARAMETERIIVEXTPROC) load(userptr, "glTextureParameterIivEXT");
    context->TextureParameterIuivEXT = (PFNGLTEXTUREPARAMETERIUIVEXTPROC) load(userptr, "glTextureParameterIuivEXT");
    context->TextureParameterfEXT = (PFNGLTEXTUREPARAMETERFEXTPROC) load(userptr, "glTextureParameterfEXT");
    context->TextureParameterfvEXT = (PFNGLTEXTUREPARAMETERFVEXTPROC) load(userptr, "glTextureParameterfvEXT");
    context->TextureParameteriEXT = (PFNGLTEXTUREPARAMETERIEXTPROC) load(userptr, "glTextureParameteriEXT");
    context->TextureParameterivEXT = (PFNGLTEXTUREPARAMETERIVEXTPROC) load(userptr, "glTextureParameterivEXT");
    context->TextureRenderbufferEXT = (PFNGLTEXTURERENDERBUFFEREXTPROC) load(userptr, "glTextureRenderbufferEXT");
    context->TextureStorage1DEXT = (PFNGLTEXTURESTORAGE1DEXTPROC) load(userptr, "glTextureStorage1DEXT");
    context->TextureStorage2DEXT = (PFNGLTEXTURESTORAGE2DEXTPROC) load(userptr, "glTextureStorage2DEXT");
    context->TextureStorage2DMultisampleEXT = (PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC) load(userptr, "glTextureStorage2DMultisampleEXT");
    context->TextureStorage3DEXT = (PFNGLTEXTURESTORAGE3DEXTPROC) load(userptr, "glTextureStorage3DEXT");
    context->TextureStorage3DMultisampleEXT = (PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC) load(userptr, "glTextureStorage3DMultisampleEXT");
    context->TextureSubImage1DEXT = (PFNGLTEXTURESUBIMAGE1DEXTPROC) load(userptr, "glTextureSubImage1DEXT");
    context->TextureSubImage2DEXT = (PFNGLTEXTURESUBIMAGE2DEXTPROC) load(userptr, "glTextureSubImage2DEXT");
    context->TextureSubImage3DEXT = (PFNGLTEXTURESUBIMAGE3DEXTPROC) load(userptr, "glTextureSubImage3DEXT");
    context->UnmapNamedBufferEXT = (PFNGLUNMAPNAMEDBUFFEREXTPROC) load(userptr, "glUnmapNamedBufferEXT");
    context->VertexArrayBindVertexBufferEXT = (PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC) load(userptr, "glVertexArrayBindVertexBufferEXT");
    context->VertexArrayColorOffsetEXT = (PFNGLVERTEXARRAYCOLOROFFSETEXTPROC) load(userptr, "glVertexArrayColorOffsetEXT");
    context->VertexArrayEdgeFlagOffsetEXT = (PFNGLVERTEXARRAYEDGEFLAGOFFSETEXTPROC) load(userptr, "glVertexArrayEdgeFlagOffsetEXT");
    context->VertexArrayFogCoordOffsetEXT = (PFNGLVERTEXARRAYFOGCOORDOFFSETEXTPROC) load(userptr, "glVertexArrayFogCoordOffsetEXT");
    context->VertexArrayIndexOffsetEXT = (PFNGLVERTEXARRAYINDEXOFFSETEXTPROC) load(userptr, "glVertexArrayIndexOffsetEXT");
    context->VertexArrayMultiTexCoordOffsetEXT = (PFNGLVERTEXARRAYMULTITEXCOORDOFFSETEXTPROC) load(userptr, "glVertexArrayMultiTexCoordOffsetEXT");
    context->VertexArrayNormalOffsetEXT = (PFNGLVERTEXARRAYNORMALOFFSETEXTPROC) load(userptr, "glVertexArrayNormalOffsetEXT");
    context->VertexArraySecondaryColorOffsetEXT = (PFNGLVERTEXARRAYSECONDARYCOLOROFFSETEXTPROC) load(userptr, "glVertexArraySecondaryColorOffsetEXT");
    context->VertexArrayTexCoordOffsetEXT = (PFNGLVERTEXARRAYTEXCOORDOFFSETEXTPROC) load(userptr, "glVertexArrayTexCoordOffsetEXT");
    context->VertexArrayVertexAttribBindingEXT = (PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC) load(userptr, "glVertexArrayVertexAttribBindingEXT");
    context->VertexArrayVertexAttribDivisorEXT = (PFNGLVERTEXARRAYVERTEXATTRIBDIVISOREXTPROC) load(userptr, "glVertexArrayVertexAttribDivisorEXT");
    context->VertexArrayVertexAttribFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC) load(userptr, "glVertexArrayVertexAttribFormatEXT");
    context->VertexArrayVertexAttribIFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC) load(userptr, "glVertexArrayVertexAttribIFormatEXT");
    context->VertexArrayVertexAttribIOffsetEXT = (PFNGLVERTEXARRAYVERTEXATTRIBIOFFSETEXTPROC) load(userptr, "glVertexArrayVertexAttribIOffsetEXT");
    context->VertexArrayVertexAttribLFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC) load(userptr, "glVertexArrayVertexAttribLFormatEXT");
    context->VertexArrayVertexAttribLOffsetEXT = (PFNGLVERTEXARRAYVERTEXATTRIBLOFFSETEXTPROC) load(userptr, "glVertexArrayVertexAttribLOffsetEXT");
    context->VertexArrayVertexAttribOffsetEXT = (PFNGLVERTEXARRAYVERTEXATTRIBOFFSETEXTPROC) load(userptr, "glVertexArrayVertexAttribOffsetEXT");
    context->VertexArrayVertexBindingDivisorEXT = (PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC) load(userptr, "glVertexArrayVertexBindingDivisorEXT");
    context->VertexArrayVertexOffsetEXT = (PFNGLVERTEXARRAYVERTEXOFFSETEXTPROC) load(userptr, "glVertexArrayVertexOffsetEXT");
}
static void glad_gl_load_GL_EXT_draw_buffers2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_draw_buffers2) return;
    context->ColorMaskIndexedEXT = (PFNGLCOLORMASKINDEXEDEXTPROC) load(userptr, "glColorMaskIndexedEXT");
    context->DisableIndexedEXT = (PFNGLDISABLEINDEXEDEXTPROC) load(userptr, "glDisableIndexedEXT");
    context->EnableIndexedEXT = (PFNGLENABLEINDEXEDEXTPROC) load(userptr, "glEnableIndexedEXT");
    context->GetBooleanIndexedvEXT = (PFNGLGETBOOLEANINDEXEDVEXTPROC) load(userptr, "glGetBooleanIndexedvEXT");
    context->GetIntegerIndexedvEXT = (PFNGLGETINTEGERINDEXEDVEXTPROC) load(userptr, "glGetIntegerIndexedvEXT");
    context->IsEnabledIndexedEXT = (PFNGLISENABLEDINDEXEDEXTPROC) load(userptr, "glIsEnabledIndexedEXT");
}
static void glad_gl_load_GL_EXT_draw_instanced(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_draw_instanced) return;
    context->DrawArraysInstancedEXT = (PFNGLDRAWARRAYSINSTANCEDEXTPROC) load(userptr, "glDrawArraysInstancedEXT");
    context->DrawElementsInstancedEXT = (PFNGLDRAWELEMENTSINSTANCEDEXTPROC) load(userptr, "glDrawElementsInstancedEXT");
}
static void glad_gl_load_GL_EXT_draw_range_elements(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_draw_range_elements) return;
    context->DrawRangeElementsEXT = (PFNGLDRAWRANGEELEMENTSEXTPROC) load(userptr, "glDrawRangeElementsEXT");
}
static void glad_gl_load_GL_EXT_external_buffer(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_external_buffer) return;
    context->BufferStorageExternalEXT = (PFNGLBUFFERSTORAGEEXTERNALEXTPROC) load(userptr, "glBufferStorageExternalEXT");
    context->NamedBufferStorageExternalEXT = (PFNGLNAMEDBUFFERSTORAGEEXTERNALEXTPROC) load(userptr, "glNamedBufferStorageExternalEXT");
}
static void glad_gl_load_GL_EXT_fog_coord(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_fog_coord) return;
    context->FogCoordPointerEXT = (PFNGLFOGCOORDPOINTEREXTPROC) load(userptr, "glFogCoordPointerEXT");
    context->FogCoorddEXT = (PFNGLFOGCOORDDEXTPROC) load(userptr, "glFogCoorddEXT");
    context->FogCoorddvEXT = (PFNGLFOGCOORDDVEXTPROC) load(userptr, "glFogCoorddvEXT");
    context->FogCoordfEXT = (PFNGLFOGCOORDFEXTPROC) load(userptr, "glFogCoordfEXT");
    context->FogCoordfvEXT = (PFNGLFOGCOORDFVEXTPROC) load(userptr, "glFogCoordfvEXT");
}
static void glad_gl_load_GL_EXT_framebuffer_blit(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_framebuffer_blit) return;
    context->BlitFramebufferEXT = (PFNGLBLITFRAMEBUFFEREXTPROC) load(userptr, "glBlitFramebufferEXT");
}
static void glad_gl_load_GL_EXT_framebuffer_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_framebuffer_multisample) return;
    context->RenderbufferStorageMultisampleEXT = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC) load(userptr, "glRenderbufferStorageMultisampleEXT");
}
static void glad_gl_load_GL_EXT_framebuffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_framebuffer_object) return;
    context->BindFramebufferEXT = (PFNGLBINDFRAMEBUFFEREXTPROC) load(userptr, "glBindFramebufferEXT");
    context->BindRenderbufferEXT = (PFNGLBINDRENDERBUFFEREXTPROC) load(userptr, "glBindRenderbufferEXT");
    context->CheckFramebufferStatusEXT = (PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC) load(userptr, "glCheckFramebufferStatusEXT");
    context->DeleteFramebuffersEXT = (PFNGLDELETEFRAMEBUFFERSEXTPROC) load(userptr, "glDeleteFramebuffersEXT");
    context->DeleteRenderbuffersEXT = (PFNGLDELETERENDERBUFFERSEXTPROC) load(userptr, "glDeleteRenderbuffersEXT");
    context->FramebufferRenderbufferEXT = (PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC) load(userptr, "glFramebufferRenderbufferEXT");
    context->FramebufferTexture1DEXT = (PFNGLFRAMEBUFFERTEXTURE1DEXTPROC) load(userptr, "glFramebufferTexture1DEXT");
    context->FramebufferTexture2DEXT = (PFNGLFRAMEBUFFERTEXTURE2DEXTPROC) load(userptr, "glFramebufferTexture2DEXT");
    context->FramebufferTexture3DEXT = (PFNGLFRAMEBUFFERTEXTURE3DEXTPROC) load(userptr, "glFramebufferTexture3DEXT");
    context->GenFramebuffersEXT = (PFNGLGENFRAMEBUFFERSEXTPROC) load(userptr, "glGenFramebuffersEXT");
    context->GenRenderbuffersEXT = (PFNGLGENRENDERBUFFERSEXTPROC) load(userptr, "glGenRenderbuffersEXT");
    context->GenerateMipmapEXT = (PFNGLGENERATEMIPMAPEXTPROC) load(userptr, "glGenerateMipmapEXT");
    context->GetFramebufferAttachmentParameterivEXT = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC) load(userptr, "glGetFramebufferAttachmentParameterivEXT");
    context->GetRenderbufferParameterivEXT = (PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC) load(userptr, "glGetRenderbufferParameterivEXT");
    context->IsFramebufferEXT = (PFNGLISFRAMEBUFFEREXTPROC) load(userptr, "glIsFramebufferEXT");
    context->IsRenderbufferEXT = (PFNGLISRENDERBUFFEREXTPROC) load(userptr, "glIsRenderbufferEXT");
    context->RenderbufferStorageEXT = (PFNGLRENDERBUFFERSTORAGEEXTPROC) load(userptr, "glRenderbufferStorageEXT");
}
static void glad_gl_load_GL_EXT_geometry_shader4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_geometry_shader4) return;
    context->ProgramParameteriEXT = (PFNGLPROGRAMPARAMETERIEXTPROC) load(userptr, "glProgramParameteriEXT");
}
static void glad_gl_load_GL_EXT_gpu_program_parameters(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_gpu_program_parameters) return;
    context->ProgramEnvParameters4fvEXT = (PFNGLPROGRAMENVPARAMETERS4FVEXTPROC) load(userptr, "glProgramEnvParameters4fvEXT");
    context->ProgramLocalParameters4fvEXT = (PFNGLPROGRAMLOCALPARAMETERS4FVEXTPROC) load(userptr, "glProgramLocalParameters4fvEXT");
}
static void glad_gl_load_GL_EXT_gpu_shader4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_gpu_shader4) return;
    context->BindFragDataLocationEXT = (PFNGLBINDFRAGDATALOCATIONEXTPROC) load(userptr, "glBindFragDataLocationEXT");
    context->GetFragDataLocationEXT = (PFNGLGETFRAGDATALOCATIONEXTPROC) load(userptr, "glGetFragDataLocationEXT");
    context->GetUniformuivEXT = (PFNGLGETUNIFORMUIVEXTPROC) load(userptr, "glGetUniformuivEXT");
    context->GetVertexAttribIivEXT = (PFNGLGETVERTEXATTRIBIIVEXTPROC) load(userptr, "glGetVertexAttribIivEXT");
    context->GetVertexAttribIuivEXT = (PFNGLGETVERTEXATTRIBIUIVEXTPROC) load(userptr, "glGetVertexAttribIuivEXT");
    context->Uniform1uiEXT = (PFNGLUNIFORM1UIEXTPROC) load(userptr, "glUniform1uiEXT");
    context->Uniform1uivEXT = (PFNGLUNIFORM1UIVEXTPROC) load(userptr, "glUniform1uivEXT");
    context->Uniform2uiEXT = (PFNGLUNIFORM2UIEXTPROC) load(userptr, "glUniform2uiEXT");
    context->Uniform2uivEXT = (PFNGLUNIFORM2UIVEXTPROC) load(userptr, "glUniform2uivEXT");
    context->Uniform3uiEXT = (PFNGLUNIFORM3UIEXTPROC) load(userptr, "glUniform3uiEXT");
    context->Uniform3uivEXT = (PFNGLUNIFORM3UIVEXTPROC) load(userptr, "glUniform3uivEXT");
    context->Uniform4uiEXT = (PFNGLUNIFORM4UIEXTPROC) load(userptr, "glUniform4uiEXT");
    context->Uniform4uivEXT = (PFNGLUNIFORM4UIVEXTPROC) load(userptr, "glUniform4uivEXT");
    context->VertexAttribI1iEXT = (PFNGLVERTEXATTRIBI1IEXTPROC) load(userptr, "glVertexAttribI1iEXT");
    context->VertexAttribI1ivEXT = (PFNGLVERTEXATTRIBI1IVEXTPROC) load(userptr, "glVertexAttribI1ivEXT");
    context->VertexAttribI1uiEXT = (PFNGLVERTEXATTRIBI1UIEXTPROC) load(userptr, "glVertexAttribI1uiEXT");
    context->VertexAttribI1uivEXT = (PFNGLVERTEXATTRIBI1UIVEXTPROC) load(userptr, "glVertexAttribI1uivEXT");
    context->VertexAttribI2iEXT = (PFNGLVERTEXATTRIBI2IEXTPROC) load(userptr, "glVertexAttribI2iEXT");
    context->VertexAttribI2ivEXT = (PFNGLVERTEXATTRIBI2IVEXTPROC) load(userptr, "glVertexAttribI2ivEXT");
    context->VertexAttribI2uiEXT = (PFNGLVERTEXATTRIBI2UIEXTPROC) load(userptr, "glVertexAttribI2uiEXT");
    context->VertexAttribI2uivEXT = (PFNGLVERTEXATTRIBI2UIVEXTPROC) load(userptr, "glVertexAttribI2uivEXT");
    context->VertexAttribI3iEXT = (PFNGLVERTEXATTRIBI3IEXTPROC) load(userptr, "glVertexAttribI3iEXT");
    context->VertexAttribI3ivEXT = (PFNGLVERTEXATTRIBI3IVEXTPROC) load(userptr, "glVertexAttribI3ivEXT");
    context->VertexAttribI3uiEXT = (PFNGLVERTEXATTRIBI3UIEXTPROC) load(userptr, "glVertexAttribI3uiEXT");
    context->VertexAttribI3uivEXT = (PFNGLVERTEXATTRIBI3UIVEXTPROC) load(userptr, "glVertexAttribI3uivEXT");
    context->VertexAttribI4bvEXT = (PFNGLVERTEXATTRIBI4BVEXTPROC) load(userptr, "glVertexAttribI4bvEXT");
    context->VertexAttribI4iEXT = (PFNGLVERTEXATTRIBI4IEXTPROC) load(userptr, "glVertexAttribI4iEXT");
    context->VertexAttribI4ivEXT = (PFNGLVERTEXATTRIBI4IVEXTPROC) load(userptr, "glVertexAttribI4ivEXT");
    context->VertexAttribI4svEXT = (PFNGLVERTEXATTRIBI4SVEXTPROC) load(userptr, "glVertexAttribI4svEXT");
    context->VertexAttribI4ubvEXT = (PFNGLVERTEXATTRIBI4UBVEXTPROC) load(userptr, "glVertexAttribI4ubvEXT");
    context->VertexAttribI4uiEXT = (PFNGLVERTEXATTRIBI4UIEXTPROC) load(userptr, "glVertexAttribI4uiEXT");
    context->VertexAttribI4uivEXT = (PFNGLVERTEXATTRIBI4UIVEXTPROC) load(userptr, "glVertexAttribI4uivEXT");
    context->VertexAttribI4usvEXT = (PFNGLVERTEXATTRIBI4USVEXTPROC) load(userptr, "glVertexAttribI4usvEXT");
    context->VertexAttribIPointerEXT = (PFNGLVERTEXATTRIBIPOINTEREXTPROC) load(userptr, "glVertexAttribIPointerEXT");
}
static void glad_gl_load_GL_EXT_histogram(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_histogram) return;
    context->GetHistogramEXT = (PFNGLGETHISTOGRAMEXTPROC) load(userptr, "glGetHistogramEXT");
    context->GetHistogramParameterfvEXT = (PFNGLGETHISTOGRAMPARAMETERFVEXTPROC) load(userptr, "glGetHistogramParameterfvEXT");
    context->GetHistogramParameterivEXT = (PFNGLGETHISTOGRAMPARAMETERIVEXTPROC) load(userptr, "glGetHistogramParameterivEXT");
    context->GetMinmaxEXT = (PFNGLGETMINMAXEXTPROC) load(userptr, "glGetMinmaxEXT");
    context->GetMinmaxParameterfvEXT = (PFNGLGETMINMAXPARAMETERFVEXTPROC) load(userptr, "glGetMinmaxParameterfvEXT");
    context->GetMinmaxParameterivEXT = (PFNGLGETMINMAXPARAMETERIVEXTPROC) load(userptr, "glGetMinmaxParameterivEXT");
    context->HistogramEXT = (PFNGLHISTOGRAMEXTPROC) load(userptr, "glHistogramEXT");
    context->MinmaxEXT = (PFNGLMINMAXEXTPROC) load(userptr, "glMinmaxEXT");
    context->ResetHistogramEXT = (PFNGLRESETHISTOGRAMEXTPROC) load(userptr, "glResetHistogramEXT");
    context->ResetMinmaxEXT = (PFNGLRESETMINMAXEXTPROC) load(userptr, "glResetMinmaxEXT");
}
static void glad_gl_load_GL_EXT_index_func(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_index_func) return;
    context->IndexFuncEXT = (PFNGLINDEXFUNCEXTPROC) load(userptr, "glIndexFuncEXT");
}
static void glad_gl_load_GL_EXT_index_material(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_index_material) return;
    context->IndexMaterialEXT = (PFNGLINDEXMATERIALEXTPROC) load(userptr, "glIndexMaterialEXT");
}
static void glad_gl_load_GL_EXT_light_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_light_texture) return;
    context->ApplyTextureEXT = (PFNGLAPPLYTEXTUREEXTPROC) load(userptr, "glApplyTextureEXT");
    context->TextureLightEXT = (PFNGLTEXTURELIGHTEXTPROC) load(userptr, "glTextureLightEXT");
    context->TextureMaterialEXT = (PFNGLTEXTUREMATERIALEXTPROC) load(userptr, "glTextureMaterialEXT");
}
static void glad_gl_load_GL_EXT_memory_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_memory_object) return;
    context->BufferStorageMemEXT = (PFNGLBUFFERSTORAGEMEMEXTPROC) load(userptr, "glBufferStorageMemEXT");
    context->CreateMemoryObjectsEXT = (PFNGLCREATEMEMORYOBJECTSEXTPROC) load(userptr, "glCreateMemoryObjectsEXT");
    context->DeleteMemoryObjectsEXT = (PFNGLDELETEMEMORYOBJECTSEXTPROC) load(userptr, "glDeleteMemoryObjectsEXT");
    context->GetMemoryObjectParameterivEXT = (PFNGLGETMEMORYOBJECTPARAMETERIVEXTPROC) load(userptr, "glGetMemoryObjectParameterivEXT");
    context->GetUnsignedBytei_vEXT = (PFNGLGETUNSIGNEDBYTEI_VEXTPROC) load(userptr, "glGetUnsignedBytei_vEXT");
    context->GetUnsignedBytevEXT = (PFNGLGETUNSIGNEDBYTEVEXTPROC) load(userptr, "glGetUnsignedBytevEXT");
    context->IsMemoryObjectEXT = (PFNGLISMEMORYOBJECTEXTPROC) load(userptr, "glIsMemoryObjectEXT");
    context->MemoryObjectParameterivEXT = (PFNGLMEMORYOBJECTPARAMETERIVEXTPROC) load(userptr, "glMemoryObjectParameterivEXT");
    context->NamedBufferStorageMemEXT = (PFNGLNAMEDBUFFERSTORAGEMEMEXTPROC) load(userptr, "glNamedBufferStorageMemEXT");
    context->TexStorageMem1DEXT = (PFNGLTEXSTORAGEMEM1DEXTPROC) load(userptr, "glTexStorageMem1DEXT");
    context->TexStorageMem2DEXT = (PFNGLTEXSTORAGEMEM2DEXTPROC) load(userptr, "glTexStorageMem2DEXT");
    context->TexStorageMem2DMultisampleEXT = (PFNGLTEXSTORAGEMEM2DMULTISAMPLEEXTPROC) load(userptr, "glTexStorageMem2DMultisampleEXT");
    context->TexStorageMem3DEXT = (PFNGLTEXSTORAGEMEM3DEXTPROC) load(userptr, "glTexStorageMem3DEXT");
    context->TexStorageMem3DMultisampleEXT = (PFNGLTEXSTORAGEMEM3DMULTISAMPLEEXTPROC) load(userptr, "glTexStorageMem3DMultisampleEXT");
    context->TextureStorageMem1DEXT = (PFNGLTEXTURESTORAGEMEM1DEXTPROC) load(userptr, "glTextureStorageMem1DEXT");
    context->TextureStorageMem2DEXT = (PFNGLTEXTURESTORAGEMEM2DEXTPROC) load(userptr, "glTextureStorageMem2DEXT");
    context->TextureStorageMem2DMultisampleEXT = (PFNGLTEXTURESTORAGEMEM2DMULTISAMPLEEXTPROC) load(userptr, "glTextureStorageMem2DMultisampleEXT");
    context->TextureStorageMem3DEXT = (PFNGLTEXTURESTORAGEMEM3DEXTPROC) load(userptr, "glTextureStorageMem3DEXT");
    context->TextureStorageMem3DMultisampleEXT = (PFNGLTEXTURESTORAGEMEM3DMULTISAMPLEEXTPROC) load(userptr, "glTextureStorageMem3DMultisampleEXT");
}
static void glad_gl_load_GL_EXT_memory_object_fd(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_memory_object_fd) return;
    context->ImportMemoryFdEXT = (PFNGLIMPORTMEMORYFDEXTPROC) load(userptr, "glImportMemoryFdEXT");
}
static void glad_gl_load_GL_EXT_memory_object_win32(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_memory_object_win32) return;
    context->ImportMemoryWin32HandleEXT = (PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC) load(userptr, "glImportMemoryWin32HandleEXT");
    context->ImportMemoryWin32NameEXT = (PFNGLIMPORTMEMORYWIN32NAMEEXTPROC) load(userptr, "glImportMemoryWin32NameEXT");
}
static void glad_gl_load_GL_EXT_multi_draw_arrays(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_multi_draw_arrays) return;
    context->MultiDrawArraysEXT = (PFNGLMULTIDRAWARRAYSEXTPROC) load(userptr, "glMultiDrawArraysEXT");
    context->MultiDrawElementsEXT = (PFNGLMULTIDRAWELEMENTSEXTPROC) load(userptr, "glMultiDrawElementsEXT");
}
static void glad_gl_load_GL_EXT_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_multisample) return;
    context->SampleMaskEXT = (PFNGLSAMPLEMASKEXTPROC) load(userptr, "glSampleMaskEXT");
    context->SamplePatternEXT = (PFNGLSAMPLEPATTERNEXTPROC) load(userptr, "glSamplePatternEXT");
}
static void glad_gl_load_GL_EXT_paletted_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_paletted_texture) return;
    context->ColorTableEXT = (PFNGLCOLORTABLEEXTPROC) load(userptr, "glColorTableEXT");
    context->GetColorTableEXT = (PFNGLGETCOLORTABLEEXTPROC) load(userptr, "glGetColorTableEXT");
    context->GetColorTableParameterfvEXT = (PFNGLGETCOLORTABLEPARAMETERFVEXTPROC) load(userptr, "glGetColorTableParameterfvEXT");
    context->GetColorTableParameterivEXT = (PFNGLGETCOLORTABLEPARAMETERIVEXTPROC) load(userptr, "glGetColorTableParameterivEXT");
}
static void glad_gl_load_GL_EXT_pixel_transform(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_pixel_transform) return;
    context->GetPixelTransformParameterfvEXT = (PFNGLGETPIXELTRANSFORMPARAMETERFVEXTPROC) load(userptr, "glGetPixelTransformParameterfvEXT");
    context->GetPixelTransformParameterivEXT = (PFNGLGETPIXELTRANSFORMPARAMETERIVEXTPROC) load(userptr, "glGetPixelTransformParameterivEXT");
    context->PixelTransformParameterfEXT = (PFNGLPIXELTRANSFORMPARAMETERFEXTPROC) load(userptr, "glPixelTransformParameterfEXT");
    context->PixelTransformParameterfvEXT = (PFNGLPIXELTRANSFORMPARAMETERFVEXTPROC) load(userptr, "glPixelTransformParameterfvEXT");
    context->PixelTransformParameteriEXT = (PFNGLPIXELTRANSFORMPARAMETERIEXTPROC) load(userptr, "glPixelTransformParameteriEXT");
    context->PixelTransformParameterivEXT = (PFNGLPIXELTRANSFORMPARAMETERIVEXTPROC) load(userptr, "glPixelTransformParameterivEXT");
}
static void glad_gl_load_GL_EXT_point_parameters(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_point_parameters) return;
    context->PointParameterfEXT = (PFNGLPOINTPARAMETERFEXTPROC) load(userptr, "glPointParameterfEXT");
    context->PointParameterfvEXT = (PFNGLPOINTPARAMETERFVEXTPROC) load(userptr, "glPointParameterfvEXT");
}
static void glad_gl_load_GL_EXT_polygon_offset(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_polygon_offset) return;
    context->PolygonOffsetEXT = (PFNGLPOLYGONOFFSETEXTPROC) load(userptr, "glPolygonOffsetEXT");
}
static void glad_gl_load_GL_EXT_polygon_offset_clamp(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_polygon_offset_clamp) return;
    context->PolygonOffsetClampEXT = (PFNGLPOLYGONOFFSETCLAMPEXTPROC) load(userptr, "glPolygonOffsetClampEXT");
}
static void glad_gl_load_GL_EXT_provoking_vertex(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_provoking_vertex) return;
    context->ProvokingVertexEXT = (PFNGLPROVOKINGVERTEXEXTPROC) load(userptr, "glProvokingVertexEXT");
}
static void glad_gl_load_GL_EXT_raster_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_raster_multisample) return;
    context->RasterSamplesEXT = (PFNGLRASTERSAMPLESEXTPROC) load(userptr, "glRasterSamplesEXT");
}
static void glad_gl_load_GL_EXT_secondary_color(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_secondary_color) return;
    context->SecondaryColor3bEXT = (PFNGLSECONDARYCOLOR3BEXTPROC) load(userptr, "glSecondaryColor3bEXT");
    context->SecondaryColor3bvEXT = (PFNGLSECONDARYCOLOR3BVEXTPROC) load(userptr, "glSecondaryColor3bvEXT");
    context->SecondaryColor3dEXT = (PFNGLSECONDARYCOLOR3DEXTPROC) load(userptr, "glSecondaryColor3dEXT");
    context->SecondaryColor3dvEXT = (PFNGLSECONDARYCOLOR3DVEXTPROC) load(userptr, "glSecondaryColor3dvEXT");
    context->SecondaryColor3fEXT = (PFNGLSECONDARYCOLOR3FEXTPROC) load(userptr, "glSecondaryColor3fEXT");
    context->SecondaryColor3fvEXT = (PFNGLSECONDARYCOLOR3FVEXTPROC) load(userptr, "glSecondaryColor3fvEXT");
    context->SecondaryColor3iEXT = (PFNGLSECONDARYCOLOR3IEXTPROC) load(userptr, "glSecondaryColor3iEXT");
    context->SecondaryColor3ivEXT = (PFNGLSECONDARYCOLOR3IVEXTPROC) load(userptr, "glSecondaryColor3ivEXT");
    context->SecondaryColor3sEXT = (PFNGLSECONDARYCOLOR3SEXTPROC) load(userptr, "glSecondaryColor3sEXT");
    context->SecondaryColor3svEXT = (PFNGLSECONDARYCOLOR3SVEXTPROC) load(userptr, "glSecondaryColor3svEXT");
    context->SecondaryColor3ubEXT = (PFNGLSECONDARYCOLOR3UBEXTPROC) load(userptr, "glSecondaryColor3ubEXT");
    context->SecondaryColor3ubvEXT = (PFNGLSECONDARYCOLOR3UBVEXTPROC) load(userptr, "glSecondaryColor3ubvEXT");
    context->SecondaryColor3uiEXT = (PFNGLSECONDARYCOLOR3UIEXTPROC) load(userptr, "glSecondaryColor3uiEXT");
    context->SecondaryColor3uivEXT = (PFNGLSECONDARYCOLOR3UIVEXTPROC) load(userptr, "glSecondaryColor3uivEXT");
    context->SecondaryColor3usEXT = (PFNGLSECONDARYCOLOR3USEXTPROC) load(userptr, "glSecondaryColor3usEXT");
    context->SecondaryColor3usvEXT = (PFNGLSECONDARYCOLOR3USVEXTPROC) load(userptr, "glSecondaryColor3usvEXT");
    context->SecondaryColorPointerEXT = (PFNGLSECONDARYCOLORPOINTEREXTPROC) load(userptr, "glSecondaryColorPointerEXT");
}
static void glad_gl_load_GL_EXT_semaphore(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_semaphore) return;
    context->DeleteSemaphoresEXT = (PFNGLDELETESEMAPHORESEXTPROC) load(userptr, "glDeleteSemaphoresEXT");
    context->GenSemaphoresEXT = (PFNGLGENSEMAPHORESEXTPROC) load(userptr, "glGenSemaphoresEXT");
    context->GetSemaphoreParameterui64vEXT = (PFNGLGETSEMAPHOREPARAMETERUI64VEXTPROC) load(userptr, "glGetSemaphoreParameterui64vEXT");
    context->GetUnsignedBytei_vEXT = (PFNGLGETUNSIGNEDBYTEI_VEXTPROC) load(userptr, "glGetUnsignedBytei_vEXT");
    context->GetUnsignedBytevEXT = (PFNGLGETUNSIGNEDBYTEVEXTPROC) load(userptr, "glGetUnsignedBytevEXT");
    context->IsSemaphoreEXT = (PFNGLISSEMAPHOREEXTPROC) load(userptr, "glIsSemaphoreEXT");
    context->SemaphoreParameterui64vEXT = (PFNGLSEMAPHOREPARAMETERUI64VEXTPROC) load(userptr, "glSemaphoreParameterui64vEXT");
    context->SignalSemaphoreEXT = (PFNGLSIGNALSEMAPHOREEXTPROC) load(userptr, "glSignalSemaphoreEXT");
    context->WaitSemaphoreEXT = (PFNGLWAITSEMAPHOREEXTPROC) load(userptr, "glWaitSemaphoreEXT");
}
static void glad_gl_load_GL_EXT_semaphore_fd(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_semaphore_fd) return;
    context->ImportSemaphoreFdEXT = (PFNGLIMPORTSEMAPHOREFDEXTPROC) load(userptr, "glImportSemaphoreFdEXT");
}
static void glad_gl_load_GL_EXT_semaphore_win32(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_semaphore_win32) return;
    context->ImportSemaphoreWin32HandleEXT = (PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC) load(userptr, "glImportSemaphoreWin32HandleEXT");
    context->ImportSemaphoreWin32NameEXT = (PFNGLIMPORTSEMAPHOREWIN32NAMEEXTPROC) load(userptr, "glImportSemaphoreWin32NameEXT");
}
static void glad_gl_load_GL_EXT_separate_shader_objects(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_separate_shader_objects) return;
    context->ActiveProgramEXT = (PFNGLACTIVEPROGRAMEXTPROC) load(userptr, "glActiveProgramEXT");
    context->CreateShaderProgramEXT = (PFNGLCREATESHADERPROGRAMEXTPROC) load(userptr, "glCreateShaderProgramEXT");
    context->UseShaderProgramEXT = (PFNGLUSESHADERPROGRAMEXTPROC) load(userptr, "glUseShaderProgramEXT");
}
static void glad_gl_load_GL_EXT_shader_framebuffer_fetch_non_coherent(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_shader_framebuffer_fetch_non_coherent) return;
    context->FramebufferFetchBarrierEXT = (PFNGLFRAMEBUFFERFETCHBARRIEREXTPROC) load(userptr, "glFramebufferFetchBarrierEXT");
}
static void glad_gl_load_GL_EXT_shader_image_load_store(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_shader_image_load_store) return;
    context->BindImageTextureEXT = (PFNGLBINDIMAGETEXTUREEXTPROC) load(userptr, "glBindImageTextureEXT");
    context->MemoryBarrierEXT = (PFNGLMEMORYBARRIEREXTPROC) load(userptr, "glMemoryBarrierEXT");
}
static void glad_gl_load_GL_EXT_stencil_clear_tag(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_stencil_clear_tag) return;
    context->StencilClearTagEXT = (PFNGLSTENCILCLEARTAGEXTPROC) load(userptr, "glStencilClearTagEXT");
}
static void glad_gl_load_GL_EXT_stencil_two_side(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_stencil_two_side) return;
    context->ActiveStencilFaceEXT = (PFNGLACTIVESTENCILFACEEXTPROC) load(userptr, "glActiveStencilFaceEXT");
}
static void glad_gl_load_GL_EXT_subtexture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_subtexture) return;
    context->TexSubImage1DEXT = (PFNGLTEXSUBIMAGE1DEXTPROC) load(userptr, "glTexSubImage1DEXT");
    context->TexSubImage2DEXT = (PFNGLTEXSUBIMAGE2DEXTPROC) load(userptr, "glTexSubImage2DEXT");
}
static void glad_gl_load_GL_EXT_texture3D(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_texture3D) return;
    context->TexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC) load(userptr, "glTexImage3DEXT");
    context->TexSubImage3DEXT = (PFNGLTEXSUBIMAGE3DEXTPROC) load(userptr, "glTexSubImage3DEXT");
}
static void glad_gl_load_GL_EXT_texture_array(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_texture_array) return;
    context->FramebufferTextureLayerEXT = (PFNGLFRAMEBUFFERTEXTURELAYEREXTPROC) load(userptr, "glFramebufferTextureLayerEXT");
}
static void glad_gl_load_GL_EXT_texture_buffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_texture_buffer_object) return;
    context->TexBufferEXT = (PFNGLTEXBUFFEREXTPROC) load(userptr, "glTexBufferEXT");
}
static void glad_gl_load_GL_EXT_texture_integer(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_texture_integer) return;
    context->ClearColorIiEXT = (PFNGLCLEARCOLORIIEXTPROC) load(userptr, "glClearColorIiEXT");
    context->ClearColorIuiEXT = (PFNGLCLEARCOLORIUIEXTPROC) load(userptr, "glClearColorIuiEXT");
    context->GetTexParameterIivEXT = (PFNGLGETTEXPARAMETERIIVEXTPROC) load(userptr, "glGetTexParameterIivEXT");
    context->GetTexParameterIuivEXT = (PFNGLGETTEXPARAMETERIUIVEXTPROC) load(userptr, "glGetTexParameterIuivEXT");
    context->TexParameterIivEXT = (PFNGLTEXPARAMETERIIVEXTPROC) load(userptr, "glTexParameterIivEXT");
    context->TexParameterIuivEXT = (PFNGLTEXPARAMETERIUIVEXTPROC) load(userptr, "glTexParameterIuivEXT");
}
static void glad_gl_load_GL_EXT_texture_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_texture_object) return;
    context->AreTexturesResidentEXT = (PFNGLARETEXTURESRESIDENTEXTPROC) load(userptr, "glAreTexturesResidentEXT");
    context->BindTextureEXT = (PFNGLBINDTEXTUREEXTPROC) load(userptr, "glBindTextureEXT");
    context->DeleteTexturesEXT = (PFNGLDELETETEXTURESEXTPROC) load(userptr, "glDeleteTexturesEXT");
    context->GenTexturesEXT = (PFNGLGENTEXTURESEXTPROC) load(userptr, "glGenTexturesEXT");
    context->IsTextureEXT = (PFNGLISTEXTUREEXTPROC) load(userptr, "glIsTextureEXT");
    context->PrioritizeTexturesEXT = (PFNGLPRIORITIZETEXTURESEXTPROC) load(userptr, "glPrioritizeTexturesEXT");
}
static void glad_gl_load_GL_EXT_texture_perturb_normal(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_texture_perturb_normal) return;
    context->TextureNormalEXT = (PFNGLTEXTURENORMALEXTPROC) load(userptr, "glTextureNormalEXT");
}
static void glad_gl_load_GL_EXT_texture_storage(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_texture_storage) return;
    context->TexStorage1DEXT = (PFNGLTEXSTORAGE1DEXTPROC) load(userptr, "glTexStorage1DEXT");
    context->TexStorage2DEXT = (PFNGLTEXSTORAGE2DEXTPROC) load(userptr, "glTexStorage2DEXT");
    context->TexStorage3DEXT = (PFNGLTEXSTORAGE3DEXTPROC) load(userptr, "glTexStorage3DEXT");
    context->TextureStorage1DEXT = (PFNGLTEXTURESTORAGE1DEXTPROC) load(userptr, "glTextureStorage1DEXT");
    context->TextureStorage2DEXT = (PFNGLTEXTURESTORAGE2DEXTPROC) load(userptr, "glTextureStorage2DEXT");
    context->TextureStorage3DEXT = (PFNGLTEXTURESTORAGE3DEXTPROC) load(userptr, "glTextureStorage3DEXT");
}
static void glad_gl_load_GL_EXT_timer_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_timer_query) return;
    context->GetQueryObjecti64vEXT = (PFNGLGETQUERYOBJECTI64VEXTPROC) load(userptr, "glGetQueryObjecti64vEXT");
    context->GetQueryObjectui64vEXT = (PFNGLGETQUERYOBJECTUI64VEXTPROC) load(userptr, "glGetQueryObjectui64vEXT");
}
static void glad_gl_load_GL_EXT_transform_feedback(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_transform_feedback) return;
    context->BeginTransformFeedbackEXT = (PFNGLBEGINTRANSFORMFEEDBACKEXTPROC) load(userptr, "glBeginTransformFeedbackEXT");
    context->BindBufferBaseEXT = (PFNGLBINDBUFFERBASEEXTPROC) load(userptr, "glBindBufferBaseEXT");
    context->BindBufferOffsetEXT = (PFNGLBINDBUFFEROFFSETEXTPROC) load(userptr, "glBindBufferOffsetEXT");
    context->BindBufferRangeEXT = (PFNGLBINDBUFFERRANGEEXTPROC) load(userptr, "glBindBufferRangeEXT");
    context->EndTransformFeedbackEXT = (PFNGLENDTRANSFORMFEEDBACKEXTPROC) load(userptr, "glEndTransformFeedbackEXT");
    context->GetTransformFeedbackVaryingEXT = (PFNGLGETTRANSFORMFEEDBACKVARYINGEXTPROC) load(userptr, "glGetTransformFeedbackVaryingEXT");
    context->TransformFeedbackVaryingsEXT = (PFNGLTRANSFORMFEEDBACKVARYINGSEXTPROC) load(userptr, "glTransformFeedbackVaryingsEXT");
}
static void glad_gl_load_GL_EXT_vertex_array(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_vertex_array) return;
    context->ArrayElementEXT = (PFNGLARRAYELEMENTEXTPROC) load(userptr, "glArrayElementEXT");
    context->ColorPointerEXT = (PFNGLCOLORPOINTEREXTPROC) load(userptr, "glColorPointerEXT");
    context->DrawArraysEXT = (PFNGLDRAWARRAYSEXTPROC) load(userptr, "glDrawArraysEXT");
    context->EdgeFlagPointerEXT = (PFNGLEDGEFLAGPOINTEREXTPROC) load(userptr, "glEdgeFlagPointerEXT");
    context->GetPointervEXT = (PFNGLGETPOINTERVEXTPROC) load(userptr, "glGetPointervEXT");
    context->IndexPointerEXT = (PFNGLINDEXPOINTEREXTPROC) load(userptr, "glIndexPointerEXT");
    context->NormalPointerEXT = (PFNGLNORMALPOINTEREXTPROC) load(userptr, "glNormalPointerEXT");
    context->TexCoordPointerEXT = (PFNGLTEXCOORDPOINTEREXTPROC) load(userptr, "glTexCoordPointerEXT");
    context->VertexPointerEXT = (PFNGLVERTEXPOINTEREXTPROC) load(userptr, "glVertexPointerEXT");
}
static void glad_gl_load_GL_EXT_vertex_attrib_64bit(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_vertex_attrib_64bit) return;
    context->GetVertexAttribLdvEXT = (PFNGLGETVERTEXATTRIBLDVEXTPROC) load(userptr, "glGetVertexAttribLdvEXT");
    context->VertexAttribL1dEXT = (PFNGLVERTEXATTRIBL1DEXTPROC) load(userptr, "glVertexAttribL1dEXT");
    context->VertexAttribL1dvEXT = (PFNGLVERTEXATTRIBL1DVEXTPROC) load(userptr, "glVertexAttribL1dvEXT");
    context->VertexAttribL2dEXT = (PFNGLVERTEXATTRIBL2DEXTPROC) load(userptr, "glVertexAttribL2dEXT");
    context->VertexAttribL2dvEXT = (PFNGLVERTEXATTRIBL2DVEXTPROC) load(userptr, "glVertexAttribL2dvEXT");
    context->VertexAttribL3dEXT = (PFNGLVERTEXATTRIBL3DEXTPROC) load(userptr, "glVertexAttribL3dEXT");
    context->VertexAttribL3dvEXT = (PFNGLVERTEXATTRIBL3DVEXTPROC) load(userptr, "glVertexAttribL3dvEXT");
    context->VertexAttribL4dEXT = (PFNGLVERTEXATTRIBL4DEXTPROC) load(userptr, "glVertexAttribL4dEXT");
    context->VertexAttribL4dvEXT = (PFNGLVERTEXATTRIBL4DVEXTPROC) load(userptr, "glVertexAttribL4dvEXT");
    context->VertexAttribLPointerEXT = (PFNGLVERTEXATTRIBLPOINTEREXTPROC) load(userptr, "glVertexAttribLPointerEXT");
}
static void glad_gl_load_GL_EXT_vertex_shader(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_vertex_shader) return;
    context->BeginVertexShaderEXT = (PFNGLBEGINVERTEXSHADEREXTPROC) load(userptr, "glBeginVertexShaderEXT");
    context->BindLightParameterEXT = (PFNGLBINDLIGHTPARAMETEREXTPROC) load(userptr, "glBindLightParameterEXT");
    context->BindMaterialParameterEXT = (PFNGLBINDMATERIALPARAMETEREXTPROC) load(userptr, "glBindMaterialParameterEXT");
    context->BindParameterEXT = (PFNGLBINDPARAMETEREXTPROC) load(userptr, "glBindParameterEXT");
    context->BindTexGenParameterEXT = (PFNGLBINDTEXGENPARAMETEREXTPROC) load(userptr, "glBindTexGenParameterEXT");
    context->BindTextureUnitParameterEXT = (PFNGLBINDTEXTUREUNITPARAMETEREXTPROC) load(userptr, "glBindTextureUnitParameterEXT");
    context->BindVertexShaderEXT = (PFNGLBINDVERTEXSHADEREXTPROC) load(userptr, "glBindVertexShaderEXT");
    context->DeleteVertexShaderEXT = (PFNGLDELETEVERTEXSHADEREXTPROC) load(userptr, "glDeleteVertexShaderEXT");
    context->DisableVariantClientStateEXT = (PFNGLDISABLEVARIANTCLIENTSTATEEXTPROC) load(userptr, "glDisableVariantClientStateEXT");
    context->EnableVariantClientStateEXT = (PFNGLENABLEVARIANTCLIENTSTATEEXTPROC) load(userptr, "glEnableVariantClientStateEXT");
    context->EndVertexShaderEXT = (PFNGLENDVERTEXSHADEREXTPROC) load(userptr, "glEndVertexShaderEXT");
    context->ExtractComponentEXT = (PFNGLEXTRACTCOMPONENTEXTPROC) load(userptr, "glExtractComponentEXT");
    context->GenSymbolsEXT = (PFNGLGENSYMBOLSEXTPROC) load(userptr, "glGenSymbolsEXT");
    context->GenVertexShadersEXT = (PFNGLGENVERTEXSHADERSEXTPROC) load(userptr, "glGenVertexShadersEXT");
    context->GetInvariantBooleanvEXT = (PFNGLGETINVARIANTBOOLEANVEXTPROC) load(userptr, "glGetInvariantBooleanvEXT");
    context->GetInvariantFloatvEXT = (PFNGLGETINVARIANTFLOATVEXTPROC) load(userptr, "glGetInvariantFloatvEXT");
    context->GetInvariantIntegervEXT = (PFNGLGETINVARIANTINTEGERVEXTPROC) load(userptr, "glGetInvariantIntegervEXT");
    context->GetLocalConstantBooleanvEXT = (PFNGLGETLOCALCONSTANTBOOLEANVEXTPROC) load(userptr, "glGetLocalConstantBooleanvEXT");
    context->GetLocalConstantFloatvEXT = (PFNGLGETLOCALCONSTANTFLOATVEXTPROC) load(userptr, "glGetLocalConstantFloatvEXT");
    context->GetLocalConstantIntegervEXT = (PFNGLGETLOCALCONSTANTINTEGERVEXTPROC) load(userptr, "glGetLocalConstantIntegervEXT");
    context->GetVariantBooleanvEXT = (PFNGLGETVARIANTBOOLEANVEXTPROC) load(userptr, "glGetVariantBooleanvEXT");
    context->GetVariantFloatvEXT = (PFNGLGETVARIANTFLOATVEXTPROC) load(userptr, "glGetVariantFloatvEXT");
    context->GetVariantIntegervEXT = (PFNGLGETVARIANTINTEGERVEXTPROC) load(userptr, "glGetVariantIntegervEXT");
    context->GetVariantPointervEXT = (PFNGLGETVARIANTPOINTERVEXTPROC) load(userptr, "glGetVariantPointervEXT");
    context->InsertComponentEXT = (PFNGLINSERTCOMPONENTEXTPROC) load(userptr, "glInsertComponentEXT");
    context->IsVariantEnabledEXT = (PFNGLISVARIANTENABLEDEXTPROC) load(userptr, "glIsVariantEnabledEXT");
    context->SetInvariantEXT = (PFNGLSETINVARIANTEXTPROC) load(userptr, "glSetInvariantEXT");
    context->SetLocalConstantEXT = (PFNGLSETLOCALCONSTANTEXTPROC) load(userptr, "glSetLocalConstantEXT");
    context->ShaderOp1EXT = (PFNGLSHADEROP1EXTPROC) load(userptr, "glShaderOp1EXT");
    context->ShaderOp2EXT = (PFNGLSHADEROP2EXTPROC) load(userptr, "glShaderOp2EXT");
    context->ShaderOp3EXT = (PFNGLSHADEROP3EXTPROC) load(userptr, "glShaderOp3EXT");
    context->SwizzleEXT = (PFNGLSWIZZLEEXTPROC) load(userptr, "glSwizzleEXT");
    context->VariantPointerEXT = (PFNGLVARIANTPOINTEREXTPROC) load(userptr, "glVariantPointerEXT");
    context->VariantbvEXT = (PFNGLVARIANTBVEXTPROC) load(userptr, "glVariantbvEXT");
    context->VariantdvEXT = (PFNGLVARIANTDVEXTPROC) load(userptr, "glVariantdvEXT");
    context->VariantfvEXT = (PFNGLVARIANTFVEXTPROC) load(userptr, "glVariantfvEXT");
    context->VariantivEXT = (PFNGLVARIANTIVEXTPROC) load(userptr, "glVariantivEXT");
    context->VariantsvEXT = (PFNGLVARIANTSVEXTPROC) load(userptr, "glVariantsvEXT");
    context->VariantubvEXT = (PFNGLVARIANTUBVEXTPROC) load(userptr, "glVariantubvEXT");
    context->VariantuivEXT = (PFNGLVARIANTUIVEXTPROC) load(userptr, "glVariantuivEXT");
    context->VariantusvEXT = (PFNGLVARIANTUSVEXTPROC) load(userptr, "glVariantusvEXT");
    context->WriteMaskEXT = (PFNGLWRITEMASKEXTPROC) load(userptr, "glWriteMaskEXT");
}
static void glad_gl_load_GL_EXT_vertex_weighting(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_vertex_weighting) return;
    context->VertexWeightPointerEXT = (PFNGLVERTEXWEIGHTPOINTEREXTPROC) load(userptr, "glVertexWeightPointerEXT");
    context->VertexWeightfEXT = (PFNGLVERTEXWEIGHTFEXTPROC) load(userptr, "glVertexWeightfEXT");
    context->VertexWeightfvEXT = (PFNGLVERTEXWEIGHTFVEXTPROC) load(userptr, "glVertexWeightfvEXT");
}
static void glad_gl_load_GL_EXT_win32_keyed_mutex(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_win32_keyed_mutex) return;
    context->AcquireKeyedMutexWin32EXT = (PFNGLACQUIREKEYEDMUTEXWIN32EXTPROC) load(userptr, "glAcquireKeyedMutexWin32EXT");
    context->ReleaseKeyedMutexWin32EXT = (PFNGLRELEASEKEYEDMUTEXWIN32EXTPROC) load(userptr, "glReleaseKeyedMutexWin32EXT");
}
static void glad_gl_load_GL_EXT_window_rectangles(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_window_rectangles) return;
    context->WindowRectanglesEXT = (PFNGLWINDOWRECTANGLESEXTPROC) load(userptr, "glWindowRectanglesEXT");
}
static void glad_gl_load_GL_EXT_x11_sync_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->EXT_x11_sync_object) return;
    context->ImportSyncEXT = (PFNGLIMPORTSYNCEXTPROC) load(userptr, "glImportSyncEXT");
}
static void glad_gl_load_GL_GREMEDY_frame_terminator(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->GREMEDY_frame_terminator) return;
    context->FrameTerminatorGREMEDY = (PFNGLFRAMETERMINATORGREMEDYPROC) load(userptr, "glFrameTerminatorGREMEDY");
}
static void glad_gl_load_GL_GREMEDY_string_marker(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->GREMEDY_string_marker) return;
    context->StringMarkerGREMEDY = (PFNGLSTRINGMARKERGREMEDYPROC) load(userptr, "glStringMarkerGREMEDY");
}
static void glad_gl_load_GL_HP_image_transform(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->HP_image_transform) return;
    context->GetImageTransformParameterfvHP = (PFNGLGETIMAGETRANSFORMPARAMETERFVHPPROC) load(userptr, "glGetImageTransformParameterfvHP");
    context->GetImageTransformParameterivHP = (PFNGLGETIMAGETRANSFORMPARAMETERIVHPPROC) load(userptr, "glGetImageTransformParameterivHP");
    context->ImageTransformParameterfHP = (PFNGLIMAGETRANSFORMPARAMETERFHPPROC) load(userptr, "glImageTransformParameterfHP");
    context->ImageTransformParameterfvHP = (PFNGLIMAGETRANSFORMPARAMETERFVHPPROC) load(userptr, "glImageTransformParameterfvHP");
    context->ImageTransformParameteriHP = (PFNGLIMAGETRANSFORMPARAMETERIHPPROC) load(userptr, "glImageTransformParameteriHP");
    context->ImageTransformParameterivHP = (PFNGLIMAGETRANSFORMPARAMETERIVHPPROC) load(userptr, "glImageTransformParameterivHP");
}
static void glad_gl_load_GL_IBM_multimode_draw_arrays(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->IBM_multimode_draw_arrays) return;
    context->MultiModeDrawArraysIBM = (PFNGLMULTIMODEDRAWARRAYSIBMPROC) load(userptr, "glMultiModeDrawArraysIBM");
    context->MultiModeDrawElementsIBM = (PFNGLMULTIMODEDRAWELEMENTSIBMPROC) load(userptr, "glMultiModeDrawElementsIBM");
}
static void glad_gl_load_GL_IBM_static_data(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->IBM_static_data) return;
    context->FlushStaticDataIBM = (PFNGLFLUSHSTATICDATAIBMPROC) load(userptr, "glFlushStaticDataIBM");
}
static void glad_gl_load_GL_IBM_vertex_array_lists(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->IBM_vertex_array_lists) return;
    context->ColorPointerListIBM = (PFNGLCOLORPOINTERLISTIBMPROC) load(userptr, "glColorPointerListIBM");
    context->EdgeFlagPointerListIBM = (PFNGLEDGEFLAGPOINTERLISTIBMPROC) load(userptr, "glEdgeFlagPointerListIBM");
    context->FogCoordPointerListIBM = (PFNGLFOGCOORDPOINTERLISTIBMPROC) load(userptr, "glFogCoordPointerListIBM");
    context->IndexPointerListIBM = (PFNGLINDEXPOINTERLISTIBMPROC) load(userptr, "glIndexPointerListIBM");
    context->NormalPointerListIBM = (PFNGLNORMALPOINTERLISTIBMPROC) load(userptr, "glNormalPointerListIBM");
    context->SecondaryColorPointerListIBM = (PFNGLSECONDARYCOLORPOINTERLISTIBMPROC) load(userptr, "glSecondaryColorPointerListIBM");
    context->TexCoordPointerListIBM = (PFNGLTEXCOORDPOINTERLISTIBMPROC) load(userptr, "glTexCoordPointerListIBM");
    context->VertexPointerListIBM = (PFNGLVERTEXPOINTERLISTIBMPROC) load(userptr, "glVertexPointerListIBM");
}
static void glad_gl_load_GL_INGR_blend_func_separate(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->INGR_blend_func_separate) return;
    context->BlendFuncSeparateINGR = (PFNGLBLENDFUNCSEPARATEINGRPROC) load(userptr, "glBlendFuncSeparateINGR");
}
static void glad_gl_load_GL_INTEL_framebuffer_CMAA(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->INTEL_framebuffer_CMAA) return;
    context->ApplyFramebufferAttachmentCMAAINTEL = (PFNGLAPPLYFRAMEBUFFERATTACHMENTCMAAINTELPROC) load(userptr, "glApplyFramebufferAttachmentCMAAINTEL");
}
static void glad_gl_load_GL_INTEL_map_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->INTEL_map_texture) return;
    context->MapTexture2DINTEL = (PFNGLMAPTEXTURE2DINTELPROC) load(userptr, "glMapTexture2DINTEL");
    context->SyncTextureINTEL = (PFNGLSYNCTEXTUREINTELPROC) load(userptr, "glSyncTextureINTEL");
    context->UnmapTexture2DINTEL = (PFNGLUNMAPTEXTURE2DINTELPROC) load(userptr, "glUnmapTexture2DINTEL");
}
static void glad_gl_load_GL_INTEL_parallel_arrays(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->INTEL_parallel_arrays) return;
    context->ColorPointervINTEL = (PFNGLCOLORPOINTERVINTELPROC) load(userptr, "glColorPointervINTEL");
    context->NormalPointervINTEL = (PFNGLNORMALPOINTERVINTELPROC) load(userptr, "glNormalPointervINTEL");
    context->TexCoordPointervINTEL = (PFNGLTEXCOORDPOINTERVINTELPROC) load(userptr, "glTexCoordPointervINTEL");
    context->VertexPointervINTEL = (PFNGLVERTEXPOINTERVINTELPROC) load(userptr, "glVertexPointervINTEL");
}
static void glad_gl_load_GL_INTEL_performance_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->INTEL_performance_query) return;
    context->BeginPerfQueryINTEL = (PFNGLBEGINPERFQUERYINTELPROC) load(userptr, "glBeginPerfQueryINTEL");
    context->CreatePerfQueryINTEL = (PFNGLCREATEPERFQUERYINTELPROC) load(userptr, "glCreatePerfQueryINTEL");
    context->DeletePerfQueryINTEL = (PFNGLDELETEPERFQUERYINTELPROC) load(userptr, "glDeletePerfQueryINTEL");
    context->EndPerfQueryINTEL = (PFNGLENDPERFQUERYINTELPROC) load(userptr, "glEndPerfQueryINTEL");
    context->GetFirstPerfQueryIdINTEL = (PFNGLGETFIRSTPERFQUERYIDINTELPROC) load(userptr, "glGetFirstPerfQueryIdINTEL");
    context->GetNextPerfQueryIdINTEL = (PFNGLGETNEXTPERFQUERYIDINTELPROC) load(userptr, "glGetNextPerfQueryIdINTEL");
    context->GetPerfCounterInfoINTEL = (PFNGLGETPERFCOUNTERINFOINTELPROC) load(userptr, "glGetPerfCounterInfoINTEL");
    context->GetPerfQueryDataINTEL = (PFNGLGETPERFQUERYDATAINTELPROC) load(userptr, "glGetPerfQueryDataINTEL");
    context->GetPerfQueryIdByNameINTEL = (PFNGLGETPERFQUERYIDBYNAMEINTELPROC) load(userptr, "glGetPerfQueryIdByNameINTEL");
    context->GetPerfQueryInfoINTEL = (PFNGLGETPERFQUERYINFOINTELPROC) load(userptr, "glGetPerfQueryInfoINTEL");
}
static void glad_gl_load_GL_KHR_blend_equation_advanced(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->KHR_blend_equation_advanced) return;
    context->BlendBarrierKHR = (PFNGLBLENDBARRIERKHRPROC) load(userptr, "glBlendBarrierKHR");
}
static void glad_gl_load_GL_KHR_debug(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->KHR_debug) return;
    context->DebugMessageCallback = (PFNGLDEBUGMESSAGECALLBACKPROC) load(userptr, "glDebugMessageCallback");
    context->DebugMessageControl = (PFNGLDEBUGMESSAGECONTROLPROC) load(userptr, "glDebugMessageControl");
    context->DebugMessageInsert = (PFNGLDEBUGMESSAGEINSERTPROC) load(userptr, "glDebugMessageInsert");
    context->GetDebugMessageLog = (PFNGLGETDEBUGMESSAGELOGPROC) load(userptr, "glGetDebugMessageLog");
    context->GetObjectLabel = (PFNGLGETOBJECTLABELPROC) load(userptr, "glGetObjectLabel");
    context->GetObjectPtrLabel = (PFNGLGETOBJECTPTRLABELPROC) load(userptr, "glGetObjectPtrLabel");
    context->GetPointerv = (PFNGLGETPOINTERVPROC) load(userptr, "glGetPointerv");
    context->ObjectLabel = (PFNGLOBJECTLABELPROC) load(userptr, "glObjectLabel");
    context->ObjectPtrLabel = (PFNGLOBJECTPTRLABELPROC) load(userptr, "glObjectPtrLabel");
    context->PopDebugGroup = (PFNGLPOPDEBUGGROUPPROC) load(userptr, "glPopDebugGroup");
    context->PushDebugGroup = (PFNGLPUSHDEBUGGROUPPROC) load(userptr, "glPushDebugGroup");
}
static void glad_gl_load_GL_KHR_parallel_shader_compile(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->KHR_parallel_shader_compile) return;
    context->MaxShaderCompilerThreadsKHR = (PFNGLMAXSHADERCOMPILERTHREADSKHRPROC) load(userptr, "glMaxShaderCompilerThreadsKHR");
}
static void glad_gl_load_GL_KHR_robustness(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->KHR_robustness) return;
    context->GetGraphicsResetStatus = (PFNGLGETGRAPHICSRESETSTATUSPROC) load(userptr, "glGetGraphicsResetStatus");
    context->GetnUniformfv = (PFNGLGETNUNIFORMFVPROC) load(userptr, "glGetnUniformfv");
    context->GetnUniformiv = (PFNGLGETNUNIFORMIVPROC) load(userptr, "glGetnUniformiv");
    context->GetnUniformuiv = (PFNGLGETNUNIFORMUIVPROC) load(userptr, "glGetnUniformuiv");
    context->ReadnPixels = (PFNGLREADNPIXELSPROC) load(userptr, "glReadnPixels");
}
static void glad_gl_load_GL_MESA_framebuffer_flip_y(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->MESA_framebuffer_flip_y) return;
    context->FramebufferParameteriMESA = (PFNGLFRAMEBUFFERPARAMETERIMESAPROC) load(userptr, "glFramebufferParameteriMESA");
    context->GetFramebufferParameterivMESA = (PFNGLGETFRAMEBUFFERPARAMETERIVMESAPROC) load(userptr, "glGetFramebufferParameterivMESA");
}
static void glad_gl_load_GL_MESA_resize_buffers(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->MESA_resize_buffers) return;
    context->ResizeBuffersMESA = (PFNGLRESIZEBUFFERSMESAPROC) load(userptr, "glResizeBuffersMESA");
}
static void glad_gl_load_GL_MESA_window_pos(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->MESA_window_pos) return;
    context->WindowPos2dMESA = (PFNGLWINDOWPOS2DMESAPROC) load(userptr, "glWindowPos2dMESA");
    context->WindowPos2dvMESA = (PFNGLWINDOWPOS2DVMESAPROC) load(userptr, "glWindowPos2dvMESA");
    context->WindowPos2fMESA = (PFNGLWINDOWPOS2FMESAPROC) load(userptr, "glWindowPos2fMESA");
    context->WindowPos2fvMESA = (PFNGLWINDOWPOS2FVMESAPROC) load(userptr, "glWindowPos2fvMESA");
    context->WindowPos2iMESA = (PFNGLWINDOWPOS2IMESAPROC) load(userptr, "glWindowPos2iMESA");
    context->WindowPos2ivMESA = (PFNGLWINDOWPOS2IVMESAPROC) load(userptr, "glWindowPos2ivMESA");
    context->WindowPos2sMESA = (PFNGLWINDOWPOS2SMESAPROC) load(userptr, "glWindowPos2sMESA");
    context->WindowPos2svMESA = (PFNGLWINDOWPOS2SVMESAPROC) load(userptr, "glWindowPos2svMESA");
    context->WindowPos3dMESA = (PFNGLWINDOWPOS3DMESAPROC) load(userptr, "glWindowPos3dMESA");
    context->WindowPos3dvMESA = (PFNGLWINDOWPOS3DVMESAPROC) load(userptr, "glWindowPos3dvMESA");
    context->WindowPos3fMESA = (PFNGLWINDOWPOS3FMESAPROC) load(userptr, "glWindowPos3fMESA");
    context->WindowPos3fvMESA = (PFNGLWINDOWPOS3FVMESAPROC) load(userptr, "glWindowPos3fvMESA");
    context->WindowPos3iMESA = (PFNGLWINDOWPOS3IMESAPROC) load(userptr, "glWindowPos3iMESA");
    context->WindowPos3ivMESA = (PFNGLWINDOWPOS3IVMESAPROC) load(userptr, "glWindowPos3ivMESA");
    context->WindowPos3sMESA = (PFNGLWINDOWPOS3SMESAPROC) load(userptr, "glWindowPos3sMESA");
    context->WindowPos3svMESA = (PFNGLWINDOWPOS3SVMESAPROC) load(userptr, "glWindowPos3svMESA");
    context->WindowPos4dMESA = (PFNGLWINDOWPOS4DMESAPROC) load(userptr, "glWindowPos4dMESA");
    context->WindowPos4dvMESA = (PFNGLWINDOWPOS4DVMESAPROC) load(userptr, "glWindowPos4dvMESA");
    context->WindowPos4fMESA = (PFNGLWINDOWPOS4FMESAPROC) load(userptr, "glWindowPos4fMESA");
    context->WindowPos4fvMESA = (PFNGLWINDOWPOS4FVMESAPROC) load(userptr, "glWindowPos4fvMESA");
    context->WindowPos4iMESA = (PFNGLWINDOWPOS4IMESAPROC) load(userptr, "glWindowPos4iMESA");
    context->WindowPos4ivMESA = (PFNGLWINDOWPOS4IVMESAPROC) load(userptr, "glWindowPos4ivMESA");
    context->WindowPos4sMESA = (PFNGLWINDOWPOS4SMESAPROC) load(userptr, "glWindowPos4sMESA");
    context->WindowPos4svMESA = (PFNGLWINDOWPOS4SVMESAPROC) load(userptr, "glWindowPos4svMESA");
}
static void glad_gl_load_GL_NVX_conditional_render(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NVX_conditional_render) return;
    context->BeginConditionalRenderNVX = (PFNGLBEGINCONDITIONALRENDERNVXPROC) load(userptr, "glBeginConditionalRenderNVX");
    context->EndConditionalRenderNVX = (PFNGLENDCONDITIONALRENDERNVXPROC) load(userptr, "glEndConditionalRenderNVX");
}
static void glad_gl_load_GL_NVX_gpu_multicast2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NVX_gpu_multicast2) return;
    context->AsyncCopyBufferSubDataNVX = (PFNGLASYNCCOPYBUFFERSUBDATANVXPROC) load(userptr, "glAsyncCopyBufferSubDataNVX");
    context->AsyncCopyImageSubDataNVX = (PFNGLASYNCCOPYIMAGESUBDATANVXPROC) load(userptr, "glAsyncCopyImageSubDataNVX");
    context->MulticastScissorArrayvNVX = (PFNGLMULTICASTSCISSORARRAYVNVXPROC) load(userptr, "glMulticastScissorArrayvNVX");
    context->MulticastViewportArrayvNVX = (PFNGLMULTICASTVIEWPORTARRAYVNVXPROC) load(userptr, "glMulticastViewportArrayvNVX");
    context->MulticastViewportPositionWScaleNVX = (PFNGLMULTICASTVIEWPORTPOSITIONWSCALENVXPROC) load(userptr, "glMulticastViewportPositionWScaleNVX");
    context->UploadGpuMaskNVX = (PFNGLUPLOADGPUMASKNVXPROC) load(userptr, "glUploadGpuMaskNVX");
}
static void glad_gl_load_GL_NVX_linked_gpu_multicast(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NVX_linked_gpu_multicast) return;
    context->LGPUCopyImageSubDataNVX = (PFNGLLGPUCOPYIMAGESUBDATANVXPROC) load(userptr, "glLGPUCopyImageSubDataNVX");
    context->LGPUInterlockNVX = (PFNGLLGPUINTERLOCKNVXPROC) load(userptr, "glLGPUInterlockNVX");
    context->LGPUNamedBufferSubDataNVX = (PFNGLLGPUNAMEDBUFFERSUBDATANVXPROC) load(userptr, "glLGPUNamedBufferSubDataNVX");
}
static void glad_gl_load_GL_NVX_progress_fence(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NVX_progress_fence) return;
    context->ClientWaitSemaphoreui64NVX = (PFNGLCLIENTWAITSEMAPHOREUI64NVXPROC) load(userptr, "glClientWaitSemaphoreui64NVX");
    context->CreateProgressFenceNVX = (PFNGLCREATEPROGRESSFENCENVXPROC) load(userptr, "glCreateProgressFenceNVX");
    context->SignalSemaphoreui64NVX = (PFNGLSIGNALSEMAPHOREUI64NVXPROC) load(userptr, "glSignalSemaphoreui64NVX");
    context->WaitSemaphoreui64NVX = (PFNGLWAITSEMAPHOREUI64NVXPROC) load(userptr, "glWaitSemaphoreui64NVX");
}
static void glad_gl_load_GL_NV_alpha_to_coverage_dither_control(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_alpha_to_coverage_dither_control) return;
    context->AlphaToCoverageDitherControlNV = (PFNGLALPHATOCOVERAGEDITHERCONTROLNVPROC) load(userptr, "glAlphaToCoverageDitherControlNV");
}
static void glad_gl_load_GL_NV_bindless_multi_draw_indirect(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_bindless_multi_draw_indirect) return;
    context->MultiDrawArraysIndirectBindlessNV = (PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSNVPROC) load(userptr, "glMultiDrawArraysIndirectBindlessNV");
    context->MultiDrawElementsIndirectBindlessNV = (PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSNVPROC) load(userptr, "glMultiDrawElementsIndirectBindlessNV");
}
static void glad_gl_load_GL_NV_bindless_multi_draw_indirect_count(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_bindless_multi_draw_indirect_count) return;
    context->MultiDrawArraysIndirectBindlessCountNV = (PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSCOUNTNVPROC) load(userptr, "glMultiDrawArraysIndirectBindlessCountNV");
    context->MultiDrawElementsIndirectBindlessCountNV = (PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSCOUNTNVPROC) load(userptr, "glMultiDrawElementsIndirectBindlessCountNV");
}
static void glad_gl_load_GL_NV_bindless_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_bindless_texture) return;
    context->GetImageHandleNV = (PFNGLGETIMAGEHANDLENVPROC) load(userptr, "glGetImageHandleNV");
    context->GetTextureHandleNV = (PFNGLGETTEXTUREHANDLENVPROC) load(userptr, "glGetTextureHandleNV");
    context->GetTextureSamplerHandleNV = (PFNGLGETTEXTURESAMPLERHANDLENVPROC) load(userptr, "glGetTextureSamplerHandleNV");
    context->IsImageHandleResidentNV = (PFNGLISIMAGEHANDLERESIDENTNVPROC) load(userptr, "glIsImageHandleResidentNV");
    context->IsTextureHandleResidentNV = (PFNGLISTEXTUREHANDLERESIDENTNVPROC) load(userptr, "glIsTextureHandleResidentNV");
    context->MakeImageHandleNonResidentNV = (PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC) load(userptr, "glMakeImageHandleNonResidentNV");
    context->MakeImageHandleResidentNV = (PFNGLMAKEIMAGEHANDLERESIDENTNVPROC) load(userptr, "glMakeImageHandleResidentNV");
    context->MakeTextureHandleNonResidentNV = (PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC) load(userptr, "glMakeTextureHandleNonResidentNV");
    context->MakeTextureHandleResidentNV = (PFNGLMAKETEXTUREHANDLERESIDENTNVPROC) load(userptr, "glMakeTextureHandleResidentNV");
    context->ProgramUniformHandleui64NV = (PFNGLPROGRAMUNIFORMHANDLEUI64NVPROC) load(userptr, "glProgramUniformHandleui64NV");
    context->ProgramUniformHandleui64vNV = (PFNGLPROGRAMUNIFORMHANDLEUI64VNVPROC) load(userptr, "glProgramUniformHandleui64vNV");
    context->UniformHandleui64NV = (PFNGLUNIFORMHANDLEUI64NVPROC) load(userptr, "glUniformHandleui64NV");
    context->UniformHandleui64vNV = (PFNGLUNIFORMHANDLEUI64VNVPROC) load(userptr, "glUniformHandleui64vNV");
}
static void glad_gl_load_GL_NV_blend_equation_advanced(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_blend_equation_advanced) return;
    context->BlendBarrierNV = (PFNGLBLENDBARRIERNVPROC) load(userptr, "glBlendBarrierNV");
    context->BlendParameteriNV = (PFNGLBLENDPARAMETERINVPROC) load(userptr, "glBlendParameteriNV");
}
static void glad_gl_load_GL_NV_clip_space_w_scaling(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_clip_space_w_scaling) return;
    context->ViewportPositionWScaleNV = (PFNGLVIEWPORTPOSITIONWSCALENVPROC) load(userptr, "glViewportPositionWScaleNV");
}
static void glad_gl_load_GL_NV_command_list(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_command_list) return;
    context->CallCommandListNV = (PFNGLCALLCOMMANDLISTNVPROC) load(userptr, "glCallCommandListNV");
    context->CommandListSegmentsNV = (PFNGLCOMMANDLISTSEGMENTSNVPROC) load(userptr, "glCommandListSegmentsNV");
    context->CompileCommandListNV = (PFNGLCOMPILECOMMANDLISTNVPROC) load(userptr, "glCompileCommandListNV");
    context->CreateCommandListsNV = (PFNGLCREATECOMMANDLISTSNVPROC) load(userptr, "glCreateCommandListsNV");
    context->CreateStatesNV = (PFNGLCREATESTATESNVPROC) load(userptr, "glCreateStatesNV");
    context->DeleteCommandListsNV = (PFNGLDELETECOMMANDLISTSNVPROC) load(userptr, "glDeleteCommandListsNV");
    context->DeleteStatesNV = (PFNGLDELETESTATESNVPROC) load(userptr, "glDeleteStatesNV");
    context->DrawCommandsAddressNV = (PFNGLDRAWCOMMANDSADDRESSNVPROC) load(userptr, "glDrawCommandsAddressNV");
    context->DrawCommandsNV = (PFNGLDRAWCOMMANDSNVPROC) load(userptr, "glDrawCommandsNV");
    context->DrawCommandsStatesAddressNV = (PFNGLDRAWCOMMANDSSTATESADDRESSNVPROC) load(userptr, "glDrawCommandsStatesAddressNV");
    context->DrawCommandsStatesNV = (PFNGLDRAWCOMMANDSSTATESNVPROC) load(userptr, "glDrawCommandsStatesNV");
    context->GetCommandHeaderNV = (PFNGLGETCOMMANDHEADERNVPROC) load(userptr, "glGetCommandHeaderNV");
    context->GetStageIndexNV = (PFNGLGETSTAGEINDEXNVPROC) load(userptr, "glGetStageIndexNV");
    context->IsCommandListNV = (PFNGLISCOMMANDLISTNVPROC) load(userptr, "glIsCommandListNV");
    context->IsStateNV = (PFNGLISSTATENVPROC) load(userptr, "glIsStateNV");
    context->ListDrawCommandsStatesClientNV = (PFNGLLISTDRAWCOMMANDSSTATESCLIENTNVPROC) load(userptr, "glListDrawCommandsStatesClientNV");
    context->StateCaptureNV = (PFNGLSTATECAPTURENVPROC) load(userptr, "glStateCaptureNV");
}
static void glad_gl_load_GL_NV_conditional_render(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_conditional_render) return;
    context->BeginConditionalRenderNV = (PFNGLBEGINCONDITIONALRENDERNVPROC) load(userptr, "glBeginConditionalRenderNV");
    context->EndConditionalRenderNV = (PFNGLENDCONDITIONALRENDERNVPROC) load(userptr, "glEndConditionalRenderNV");
}
static void glad_gl_load_GL_NV_conservative_raster(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_conservative_raster) return;
    context->SubpixelPrecisionBiasNV = (PFNGLSUBPIXELPRECISIONBIASNVPROC) load(userptr, "glSubpixelPrecisionBiasNV");
}
static void glad_gl_load_GL_NV_conservative_raster_dilate(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_conservative_raster_dilate) return;
    context->ConservativeRasterParameterfNV = (PFNGLCONSERVATIVERASTERPARAMETERFNVPROC) load(userptr, "glConservativeRasterParameterfNV");
}
static void glad_gl_load_GL_NV_conservative_raster_pre_snap_triangles(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_conservative_raster_pre_snap_triangles) return;
    context->ConservativeRasterParameteriNV = (PFNGLCONSERVATIVERASTERPARAMETERINVPROC) load(userptr, "glConservativeRasterParameteriNV");
}
static void glad_gl_load_GL_NV_copy_image(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_copy_image) return;
    context->CopyImageSubDataNV = (PFNGLCOPYIMAGESUBDATANVPROC) load(userptr, "glCopyImageSubDataNV");
}
static void glad_gl_load_GL_NV_depth_buffer_float(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_depth_buffer_float) return;
    context->ClearDepthdNV = (PFNGLCLEARDEPTHDNVPROC) load(userptr, "glClearDepthdNV");
    context->DepthBoundsdNV = (PFNGLDEPTHBOUNDSDNVPROC) load(userptr, "glDepthBoundsdNV");
    context->DepthRangedNV = (PFNGLDEPTHRANGEDNVPROC) load(userptr, "glDepthRangedNV");
}
static void glad_gl_load_GL_NV_draw_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_draw_texture) return;
    context->DrawTextureNV = (PFNGLDRAWTEXTURENVPROC) load(userptr, "glDrawTextureNV");
}
static void glad_gl_load_GL_NV_draw_vulkan_image(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_draw_vulkan_image) return;
    context->DrawVkImageNV = (PFNGLDRAWVKIMAGENVPROC) load(userptr, "glDrawVkImageNV");
    context->GetVkProcAddrNV = (PFNGLGETVKPROCADDRNVPROC) load(userptr, "glGetVkProcAddrNV");
    context->SignalVkFenceNV = (PFNGLSIGNALVKFENCENVPROC) load(userptr, "glSignalVkFenceNV");
    context->SignalVkSemaphoreNV = (PFNGLSIGNALVKSEMAPHORENVPROC) load(userptr, "glSignalVkSemaphoreNV");
    context->WaitVkSemaphoreNV = (PFNGLWAITVKSEMAPHORENVPROC) load(userptr, "glWaitVkSemaphoreNV");
}
static void glad_gl_load_GL_NV_evaluators(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_evaluators) return;
    context->EvalMapsNV = (PFNGLEVALMAPSNVPROC) load(userptr, "glEvalMapsNV");
    context->GetMapAttribParameterfvNV = (PFNGLGETMAPATTRIBPARAMETERFVNVPROC) load(userptr, "glGetMapAttribParameterfvNV");
    context->GetMapAttribParameterivNV = (PFNGLGETMAPATTRIBPARAMETERIVNVPROC) load(userptr, "glGetMapAttribParameterivNV");
    context->GetMapControlPointsNV = (PFNGLGETMAPCONTROLPOINTSNVPROC) load(userptr, "glGetMapControlPointsNV");
    context->GetMapParameterfvNV = (PFNGLGETMAPPARAMETERFVNVPROC) load(userptr, "glGetMapParameterfvNV");
    context->GetMapParameterivNV = (PFNGLGETMAPPARAMETERIVNVPROC) load(userptr, "glGetMapParameterivNV");
    context->MapControlPointsNV = (PFNGLMAPCONTROLPOINTSNVPROC) load(userptr, "glMapControlPointsNV");
    context->MapParameterfvNV = (PFNGLMAPPARAMETERFVNVPROC) load(userptr, "glMapParameterfvNV");
    context->MapParameterivNV = (PFNGLMAPPARAMETERIVNVPROC) load(userptr, "glMapParameterivNV");
}
static void glad_gl_load_GL_NV_explicit_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_explicit_multisample) return;
    context->GetMultisamplefvNV = (PFNGLGETMULTISAMPLEFVNVPROC) load(userptr, "glGetMultisamplefvNV");
    context->SampleMaskIndexedNV = (PFNGLSAMPLEMASKINDEXEDNVPROC) load(userptr, "glSampleMaskIndexedNV");
    context->TexRenderbufferNV = (PFNGLTEXRENDERBUFFERNVPROC) load(userptr, "glTexRenderbufferNV");
}
static void glad_gl_load_GL_NV_fence(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_fence) return;
    context->DeleteFencesNV = (PFNGLDELETEFENCESNVPROC) load(userptr, "glDeleteFencesNV");
    context->FinishFenceNV = (PFNGLFINISHFENCENVPROC) load(userptr, "glFinishFenceNV");
    context->GenFencesNV = (PFNGLGENFENCESNVPROC) load(userptr, "glGenFencesNV");
    context->GetFenceivNV = (PFNGLGETFENCEIVNVPROC) load(userptr, "glGetFenceivNV");
    context->IsFenceNV = (PFNGLISFENCENVPROC) load(userptr, "glIsFenceNV");
    context->SetFenceNV = (PFNGLSETFENCENVPROC) load(userptr, "glSetFenceNV");
    context->TestFenceNV = (PFNGLTESTFENCENVPROC) load(userptr, "glTestFenceNV");
}
static void glad_gl_load_GL_NV_fragment_coverage_to_color(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_fragment_coverage_to_color) return;
    context->FragmentCoverageColorNV = (PFNGLFRAGMENTCOVERAGECOLORNVPROC) load(userptr, "glFragmentCoverageColorNV");
}
static void glad_gl_load_GL_NV_fragment_program(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_fragment_program) return;
    context->GetProgramNamedParameterdvNV = (PFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC) load(userptr, "glGetProgramNamedParameterdvNV");
    context->GetProgramNamedParameterfvNV = (PFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC) load(userptr, "glGetProgramNamedParameterfvNV");
    context->ProgramNamedParameter4dNV = (PFNGLPROGRAMNAMEDPARAMETER4DNVPROC) load(userptr, "glProgramNamedParameter4dNV");
    context->ProgramNamedParameter4dvNV = (PFNGLPROGRAMNAMEDPARAMETER4DVNVPROC) load(userptr, "glProgramNamedParameter4dvNV");
    context->ProgramNamedParameter4fNV = (PFNGLPROGRAMNAMEDPARAMETER4FNVPROC) load(userptr, "glProgramNamedParameter4fNV");
    context->ProgramNamedParameter4fvNV = (PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC) load(userptr, "glProgramNamedParameter4fvNV");
}
static void glad_gl_load_GL_NV_framebuffer_mixed_samples(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_framebuffer_mixed_samples) return;
    context->CoverageModulationNV = (PFNGLCOVERAGEMODULATIONNVPROC) load(userptr, "glCoverageModulationNV");
    context->CoverageModulationTableNV = (PFNGLCOVERAGEMODULATIONTABLENVPROC) load(userptr, "glCoverageModulationTableNV");
    context->GetCoverageModulationTableNV = (PFNGLGETCOVERAGEMODULATIONTABLENVPROC) load(userptr, "glGetCoverageModulationTableNV");
    context->RasterSamplesEXT = (PFNGLRASTERSAMPLESEXTPROC) load(userptr, "glRasterSamplesEXT");
}
static void glad_gl_load_GL_NV_framebuffer_multisample_coverage(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_framebuffer_multisample_coverage) return;
    context->RenderbufferStorageMultisampleCoverageNV = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLECOVERAGENVPROC) load(userptr, "glRenderbufferStorageMultisampleCoverageNV");
}
static void glad_gl_load_GL_NV_geometry_program4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_geometry_program4) return;
    context->FramebufferTextureEXT = (PFNGLFRAMEBUFFERTEXTUREEXTPROC) load(userptr, "glFramebufferTextureEXT");
    context->FramebufferTextureFaceEXT = (PFNGLFRAMEBUFFERTEXTUREFACEEXTPROC) load(userptr, "glFramebufferTextureFaceEXT");
    context->FramebufferTextureLayerEXT = (PFNGLFRAMEBUFFERTEXTURELAYEREXTPROC) load(userptr, "glFramebufferTextureLayerEXT");
    context->ProgramVertexLimitNV = (PFNGLPROGRAMVERTEXLIMITNVPROC) load(userptr, "glProgramVertexLimitNV");
}
static void glad_gl_load_GL_NV_gpu_multicast(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_gpu_multicast) return;
    context->MulticastBarrierNV = (PFNGLMULTICASTBARRIERNVPROC) load(userptr, "glMulticastBarrierNV");
    context->MulticastBlitFramebufferNV = (PFNGLMULTICASTBLITFRAMEBUFFERNVPROC) load(userptr, "glMulticastBlitFramebufferNV");
    context->MulticastBufferSubDataNV = (PFNGLMULTICASTBUFFERSUBDATANVPROC) load(userptr, "glMulticastBufferSubDataNV");
    context->MulticastCopyBufferSubDataNV = (PFNGLMULTICASTCOPYBUFFERSUBDATANVPROC) load(userptr, "glMulticastCopyBufferSubDataNV");
    context->MulticastCopyImageSubDataNV = (PFNGLMULTICASTCOPYIMAGESUBDATANVPROC) load(userptr, "glMulticastCopyImageSubDataNV");
    context->MulticastFramebufferSampleLocationsfvNV = (PFNGLMULTICASTFRAMEBUFFERSAMPLELOCATIONSFVNVPROC) load(userptr, "glMulticastFramebufferSampleLocationsfvNV");
    context->MulticastGetQueryObjecti64vNV = (PFNGLMULTICASTGETQUERYOBJECTI64VNVPROC) load(userptr, "glMulticastGetQueryObjecti64vNV");
    context->MulticastGetQueryObjectivNV = (PFNGLMULTICASTGETQUERYOBJECTIVNVPROC) load(userptr, "glMulticastGetQueryObjectivNV");
    context->MulticastGetQueryObjectui64vNV = (PFNGLMULTICASTGETQUERYOBJECTUI64VNVPROC) load(userptr, "glMulticastGetQueryObjectui64vNV");
    context->MulticastGetQueryObjectuivNV = (PFNGLMULTICASTGETQUERYOBJECTUIVNVPROC) load(userptr, "glMulticastGetQueryObjectuivNV");
    context->MulticastWaitSyncNV = (PFNGLMULTICASTWAITSYNCNVPROC) load(userptr, "glMulticastWaitSyncNV");
    context->RenderGpuMaskNV = (PFNGLRENDERGPUMASKNVPROC) load(userptr, "glRenderGpuMaskNV");
}
static void glad_gl_load_GL_NV_gpu_program4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_gpu_program4) return;
    context->GetProgramEnvParameterIivNV = (PFNGLGETPROGRAMENVPARAMETERIIVNVPROC) load(userptr, "glGetProgramEnvParameterIivNV");
    context->GetProgramEnvParameterIuivNV = (PFNGLGETPROGRAMENVPARAMETERIUIVNVPROC) load(userptr, "glGetProgramEnvParameterIuivNV");
    context->GetProgramLocalParameterIivNV = (PFNGLGETPROGRAMLOCALPARAMETERIIVNVPROC) load(userptr, "glGetProgramLocalParameterIivNV");
    context->GetProgramLocalParameterIuivNV = (PFNGLGETPROGRAMLOCALPARAMETERIUIVNVPROC) load(userptr, "glGetProgramLocalParameterIuivNV");
    context->ProgramEnvParameterI4iNV = (PFNGLPROGRAMENVPARAMETERI4INVPROC) load(userptr, "glProgramEnvParameterI4iNV");
    context->ProgramEnvParameterI4ivNV = (PFNGLPROGRAMENVPARAMETERI4IVNVPROC) load(userptr, "glProgramEnvParameterI4ivNV");
    context->ProgramEnvParameterI4uiNV = (PFNGLPROGRAMENVPARAMETERI4UINVPROC) load(userptr, "glProgramEnvParameterI4uiNV");
    context->ProgramEnvParameterI4uivNV = (PFNGLPROGRAMENVPARAMETERI4UIVNVPROC) load(userptr, "glProgramEnvParameterI4uivNV");
    context->ProgramEnvParametersI4ivNV = (PFNGLPROGRAMENVPARAMETERSI4IVNVPROC) load(userptr, "glProgramEnvParametersI4ivNV");
    context->ProgramEnvParametersI4uivNV = (PFNGLPROGRAMENVPARAMETERSI4UIVNVPROC) load(userptr, "glProgramEnvParametersI4uivNV");
    context->ProgramLocalParameterI4iNV = (PFNGLPROGRAMLOCALPARAMETERI4INVPROC) load(userptr, "glProgramLocalParameterI4iNV");
    context->ProgramLocalParameterI4ivNV = (PFNGLPROGRAMLOCALPARAMETERI4IVNVPROC) load(userptr, "glProgramLocalParameterI4ivNV");
    context->ProgramLocalParameterI4uiNV = (PFNGLPROGRAMLOCALPARAMETERI4UINVPROC) load(userptr, "glProgramLocalParameterI4uiNV");
    context->ProgramLocalParameterI4uivNV = (PFNGLPROGRAMLOCALPARAMETERI4UIVNVPROC) load(userptr, "glProgramLocalParameterI4uivNV");
    context->ProgramLocalParametersI4ivNV = (PFNGLPROGRAMLOCALPARAMETERSI4IVNVPROC) load(userptr, "glProgramLocalParametersI4ivNV");
    context->ProgramLocalParametersI4uivNV = (PFNGLPROGRAMLOCALPARAMETERSI4UIVNVPROC) load(userptr, "glProgramLocalParametersI4uivNV");
}
static void glad_gl_load_GL_NV_gpu_program5(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_gpu_program5) return;
    context->GetProgramSubroutineParameteruivNV = (PFNGLGETPROGRAMSUBROUTINEPARAMETERUIVNVPROC) load(userptr, "glGetProgramSubroutineParameteruivNV");
    context->ProgramSubroutineParametersuivNV = (PFNGLPROGRAMSUBROUTINEPARAMETERSUIVNVPROC) load(userptr, "glProgramSubroutineParametersuivNV");
}
static void glad_gl_load_GL_NV_gpu_shader5(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_gpu_shader5) return;
    context->GetUniformi64vNV = (PFNGLGETUNIFORMI64VNVPROC) load(userptr, "glGetUniformi64vNV");
    context->ProgramUniform1i64NV = (PFNGLPROGRAMUNIFORM1I64NVPROC) load(userptr, "glProgramUniform1i64NV");
    context->ProgramUniform1i64vNV = (PFNGLPROGRAMUNIFORM1I64VNVPROC) load(userptr, "glProgramUniform1i64vNV");
    context->ProgramUniform1ui64NV = (PFNGLPROGRAMUNIFORM1UI64NVPROC) load(userptr, "glProgramUniform1ui64NV");
    context->ProgramUniform1ui64vNV = (PFNGLPROGRAMUNIFORM1UI64VNVPROC) load(userptr, "glProgramUniform1ui64vNV");
    context->ProgramUniform2i64NV = (PFNGLPROGRAMUNIFORM2I64NVPROC) load(userptr, "glProgramUniform2i64NV");
    context->ProgramUniform2i64vNV = (PFNGLPROGRAMUNIFORM2I64VNVPROC) load(userptr, "glProgramUniform2i64vNV");
    context->ProgramUniform2ui64NV = (PFNGLPROGRAMUNIFORM2UI64NVPROC) load(userptr, "glProgramUniform2ui64NV");
    context->ProgramUniform2ui64vNV = (PFNGLPROGRAMUNIFORM2UI64VNVPROC) load(userptr, "glProgramUniform2ui64vNV");
    context->ProgramUniform3i64NV = (PFNGLPROGRAMUNIFORM3I64NVPROC) load(userptr, "glProgramUniform3i64NV");
    context->ProgramUniform3i64vNV = (PFNGLPROGRAMUNIFORM3I64VNVPROC) load(userptr, "glProgramUniform3i64vNV");
    context->ProgramUniform3ui64NV = (PFNGLPROGRAMUNIFORM3UI64NVPROC) load(userptr, "glProgramUniform3ui64NV");
    context->ProgramUniform3ui64vNV = (PFNGLPROGRAMUNIFORM3UI64VNVPROC) load(userptr, "glProgramUniform3ui64vNV");
    context->ProgramUniform4i64NV = (PFNGLPROGRAMUNIFORM4I64NVPROC) load(userptr, "glProgramUniform4i64NV");
    context->ProgramUniform4i64vNV = (PFNGLPROGRAMUNIFORM4I64VNVPROC) load(userptr, "glProgramUniform4i64vNV");
    context->ProgramUniform4ui64NV = (PFNGLPROGRAMUNIFORM4UI64NVPROC) load(userptr, "glProgramUniform4ui64NV");
    context->ProgramUniform4ui64vNV = (PFNGLPROGRAMUNIFORM4UI64VNVPROC) load(userptr, "glProgramUniform4ui64vNV");
    context->Uniform1i64NV = (PFNGLUNIFORM1I64NVPROC) load(userptr, "glUniform1i64NV");
    context->Uniform1i64vNV = (PFNGLUNIFORM1I64VNVPROC) load(userptr, "glUniform1i64vNV");
    context->Uniform1ui64NV = (PFNGLUNIFORM1UI64NVPROC) load(userptr, "glUniform1ui64NV");
    context->Uniform1ui64vNV = (PFNGLUNIFORM1UI64VNVPROC) load(userptr, "glUniform1ui64vNV");
    context->Uniform2i64NV = (PFNGLUNIFORM2I64NVPROC) load(userptr, "glUniform2i64NV");
    context->Uniform2i64vNV = (PFNGLUNIFORM2I64VNVPROC) load(userptr, "glUniform2i64vNV");
    context->Uniform2ui64NV = (PFNGLUNIFORM2UI64NVPROC) load(userptr, "glUniform2ui64NV");
    context->Uniform2ui64vNV = (PFNGLUNIFORM2UI64VNVPROC) load(userptr, "glUniform2ui64vNV");
    context->Uniform3i64NV = (PFNGLUNIFORM3I64NVPROC) load(userptr, "glUniform3i64NV");
    context->Uniform3i64vNV = (PFNGLUNIFORM3I64VNVPROC) load(userptr, "glUniform3i64vNV");
    context->Uniform3ui64NV = (PFNGLUNIFORM3UI64NVPROC) load(userptr, "glUniform3ui64NV");
    context->Uniform3ui64vNV = (PFNGLUNIFORM3UI64VNVPROC) load(userptr, "glUniform3ui64vNV");
    context->Uniform4i64NV = (PFNGLUNIFORM4I64NVPROC) load(userptr, "glUniform4i64NV");
    context->Uniform4i64vNV = (PFNGLUNIFORM4I64VNVPROC) load(userptr, "glUniform4i64vNV");
    context->Uniform4ui64NV = (PFNGLUNIFORM4UI64NVPROC) load(userptr, "glUniform4ui64NV");
    context->Uniform4ui64vNV = (PFNGLUNIFORM4UI64VNVPROC) load(userptr, "glUniform4ui64vNV");
}
static void glad_gl_load_GL_NV_half_float(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_half_float) return;
    context->Color3hNV = (PFNGLCOLOR3HNVPROC) load(userptr, "glColor3hNV");
    context->Color3hvNV = (PFNGLCOLOR3HVNVPROC) load(userptr, "glColor3hvNV");
    context->Color4hNV = (PFNGLCOLOR4HNVPROC) load(userptr, "glColor4hNV");
    context->Color4hvNV = (PFNGLCOLOR4HVNVPROC) load(userptr, "glColor4hvNV");
    context->FogCoordhNV = (PFNGLFOGCOORDHNVPROC) load(userptr, "glFogCoordhNV");
    context->FogCoordhvNV = (PFNGLFOGCOORDHVNVPROC) load(userptr, "glFogCoordhvNV");
    context->MultiTexCoord1hNV = (PFNGLMULTITEXCOORD1HNVPROC) load(userptr, "glMultiTexCoord1hNV");
    context->MultiTexCoord1hvNV = (PFNGLMULTITEXCOORD1HVNVPROC) load(userptr, "glMultiTexCoord1hvNV");
    context->MultiTexCoord2hNV = (PFNGLMULTITEXCOORD2HNVPROC) load(userptr, "glMultiTexCoord2hNV");
    context->MultiTexCoord2hvNV = (PFNGLMULTITEXCOORD2HVNVPROC) load(userptr, "glMultiTexCoord2hvNV");
    context->MultiTexCoord3hNV = (PFNGLMULTITEXCOORD3HNVPROC) load(userptr, "glMultiTexCoord3hNV");
    context->MultiTexCoord3hvNV = (PFNGLMULTITEXCOORD3HVNVPROC) load(userptr, "glMultiTexCoord3hvNV");
    context->MultiTexCoord4hNV = (PFNGLMULTITEXCOORD4HNVPROC) load(userptr, "glMultiTexCoord4hNV");
    context->MultiTexCoord4hvNV = (PFNGLMULTITEXCOORD4HVNVPROC) load(userptr, "glMultiTexCoord4hvNV");
    context->Normal3hNV = (PFNGLNORMAL3HNVPROC) load(userptr, "glNormal3hNV");
    context->Normal3hvNV = (PFNGLNORMAL3HVNVPROC) load(userptr, "glNormal3hvNV");
    context->SecondaryColor3hNV = (PFNGLSECONDARYCOLOR3HNVPROC) load(userptr, "glSecondaryColor3hNV");
    context->SecondaryColor3hvNV = (PFNGLSECONDARYCOLOR3HVNVPROC) load(userptr, "glSecondaryColor3hvNV");
    context->TexCoord1hNV = (PFNGLTEXCOORD1HNVPROC) load(userptr, "glTexCoord1hNV");
    context->TexCoord1hvNV = (PFNGLTEXCOORD1HVNVPROC) load(userptr, "glTexCoord1hvNV");
    context->TexCoord2hNV = (PFNGLTEXCOORD2HNVPROC) load(userptr, "glTexCoord2hNV");
    context->TexCoord2hvNV = (PFNGLTEXCOORD2HVNVPROC) load(userptr, "glTexCoord2hvNV");
    context->TexCoord3hNV = (PFNGLTEXCOORD3HNVPROC) load(userptr, "glTexCoord3hNV");
    context->TexCoord3hvNV = (PFNGLTEXCOORD3HVNVPROC) load(userptr, "glTexCoord3hvNV");
    context->TexCoord4hNV = (PFNGLTEXCOORD4HNVPROC) load(userptr, "glTexCoord4hNV");
    context->TexCoord4hvNV = (PFNGLTEXCOORD4HVNVPROC) load(userptr, "glTexCoord4hvNV");
    context->Vertex2hNV = (PFNGLVERTEX2HNVPROC) load(userptr, "glVertex2hNV");
    context->Vertex2hvNV = (PFNGLVERTEX2HVNVPROC) load(userptr, "glVertex2hvNV");
    context->Vertex3hNV = (PFNGLVERTEX3HNVPROC) load(userptr, "glVertex3hNV");
    context->Vertex3hvNV = (PFNGLVERTEX3HVNVPROC) load(userptr, "glVertex3hvNV");
    context->Vertex4hNV = (PFNGLVERTEX4HNVPROC) load(userptr, "glVertex4hNV");
    context->Vertex4hvNV = (PFNGLVERTEX4HVNVPROC) load(userptr, "glVertex4hvNV");
    context->VertexAttrib1hNV = (PFNGLVERTEXATTRIB1HNVPROC) load(userptr, "glVertexAttrib1hNV");
    context->VertexAttrib1hvNV = (PFNGLVERTEXATTRIB1HVNVPROC) load(userptr, "glVertexAttrib1hvNV");
    context->VertexAttrib2hNV = (PFNGLVERTEXATTRIB2HNVPROC) load(userptr, "glVertexAttrib2hNV");
    context->VertexAttrib2hvNV = (PFNGLVERTEXATTRIB2HVNVPROC) load(userptr, "glVertexAttrib2hvNV");
    context->VertexAttrib3hNV = (PFNGLVERTEXATTRIB3HNVPROC) load(userptr, "glVertexAttrib3hNV");
    context->VertexAttrib3hvNV = (PFNGLVERTEXATTRIB3HVNVPROC) load(userptr, "glVertexAttrib3hvNV");
    context->VertexAttrib4hNV = (PFNGLVERTEXATTRIB4HNVPROC) load(userptr, "glVertexAttrib4hNV");
    context->VertexAttrib4hvNV = (PFNGLVERTEXATTRIB4HVNVPROC) load(userptr, "glVertexAttrib4hvNV");
    context->VertexAttribs1hvNV = (PFNGLVERTEXATTRIBS1HVNVPROC) load(userptr, "glVertexAttribs1hvNV");
    context->VertexAttribs2hvNV = (PFNGLVERTEXATTRIBS2HVNVPROC) load(userptr, "glVertexAttribs2hvNV");
    context->VertexAttribs3hvNV = (PFNGLVERTEXATTRIBS3HVNVPROC) load(userptr, "glVertexAttribs3hvNV");
    context->VertexAttribs4hvNV = (PFNGLVERTEXATTRIBS4HVNVPROC) load(userptr, "glVertexAttribs4hvNV");
    context->VertexWeighthNV = (PFNGLVERTEXWEIGHTHNVPROC) load(userptr, "glVertexWeighthNV");
    context->VertexWeighthvNV = (PFNGLVERTEXWEIGHTHVNVPROC) load(userptr, "glVertexWeighthvNV");
}
static void glad_gl_load_GL_NV_internalformat_sample_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_internalformat_sample_query) return;
    context->GetInternalformatSampleivNV = (PFNGLGETINTERNALFORMATSAMPLEIVNVPROC) load(userptr, "glGetInternalformatSampleivNV");
}
static void glad_gl_load_GL_NV_memory_attachment(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_memory_attachment) return;
    context->BufferAttachMemoryNV = (PFNGLBUFFERATTACHMEMORYNVPROC) load(userptr, "glBufferAttachMemoryNV");
    context->GetMemoryObjectDetachedResourcesuivNV = (PFNGLGETMEMORYOBJECTDETACHEDRESOURCESUIVNVPROC) load(userptr, "glGetMemoryObjectDetachedResourcesuivNV");
    context->NamedBufferAttachMemoryNV = (PFNGLNAMEDBUFFERATTACHMEMORYNVPROC) load(userptr, "glNamedBufferAttachMemoryNV");
    context->ResetMemoryObjectParameterNV = (PFNGLRESETMEMORYOBJECTPARAMETERNVPROC) load(userptr, "glResetMemoryObjectParameterNV");
    context->TexAttachMemoryNV = (PFNGLTEXATTACHMEMORYNVPROC) load(userptr, "glTexAttachMemoryNV");
    context->TextureAttachMemoryNV = (PFNGLTEXTUREATTACHMEMORYNVPROC) load(userptr, "glTextureAttachMemoryNV");
}
static void glad_gl_load_GL_NV_memory_object_sparse(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_memory_object_sparse) return;
    context->BufferPageCommitmentMemNV = (PFNGLBUFFERPAGECOMMITMENTMEMNVPROC) load(userptr, "glBufferPageCommitmentMemNV");
    context->NamedBufferPageCommitmentMemNV = (PFNGLNAMEDBUFFERPAGECOMMITMENTMEMNVPROC) load(userptr, "glNamedBufferPageCommitmentMemNV");
    context->TexPageCommitmentMemNV = (PFNGLTEXPAGECOMMITMENTMEMNVPROC) load(userptr, "glTexPageCommitmentMemNV");
    context->TexturePageCommitmentMemNV = (PFNGLTEXTUREPAGECOMMITMENTMEMNVPROC) load(userptr, "glTexturePageCommitmentMemNV");
}
static void glad_gl_load_GL_NV_mesh_shader(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_mesh_shader) return;
    context->DrawMeshTasksIndirectNV = (PFNGLDRAWMESHTASKSINDIRECTNVPROC) load(userptr, "glDrawMeshTasksIndirectNV");
    context->DrawMeshTasksNV = (PFNGLDRAWMESHTASKSNVPROC) load(userptr, "glDrawMeshTasksNV");
    context->MultiDrawMeshTasksIndirectCountNV = (PFNGLMULTIDRAWMESHTASKSINDIRECTCOUNTNVPROC) load(userptr, "glMultiDrawMeshTasksIndirectCountNV");
    context->MultiDrawMeshTasksIndirectNV = (PFNGLMULTIDRAWMESHTASKSINDIRECTNVPROC) load(userptr, "glMultiDrawMeshTasksIndirectNV");
}
static void glad_gl_load_GL_NV_occlusion_query(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_occlusion_query) return;
    context->BeginOcclusionQueryNV = (PFNGLBEGINOCCLUSIONQUERYNVPROC) load(userptr, "glBeginOcclusionQueryNV");
    context->DeleteOcclusionQueriesNV = (PFNGLDELETEOCCLUSIONQUERIESNVPROC) load(userptr, "glDeleteOcclusionQueriesNV");
    context->EndOcclusionQueryNV = (PFNGLENDOCCLUSIONQUERYNVPROC) load(userptr, "glEndOcclusionQueryNV");
    context->GenOcclusionQueriesNV = (PFNGLGENOCCLUSIONQUERIESNVPROC) load(userptr, "glGenOcclusionQueriesNV");
    context->GetOcclusionQueryivNV = (PFNGLGETOCCLUSIONQUERYIVNVPROC) load(userptr, "glGetOcclusionQueryivNV");
    context->GetOcclusionQueryuivNV = (PFNGLGETOCCLUSIONQUERYUIVNVPROC) load(userptr, "glGetOcclusionQueryuivNV");
    context->IsOcclusionQueryNV = (PFNGLISOCCLUSIONQUERYNVPROC) load(userptr, "glIsOcclusionQueryNV");
}
static void glad_gl_load_GL_NV_parameter_buffer_object(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_parameter_buffer_object) return;
    context->ProgramBufferParametersIivNV = (PFNGLPROGRAMBUFFERPARAMETERSIIVNVPROC) load(userptr, "glProgramBufferParametersIivNV");
    context->ProgramBufferParametersIuivNV = (PFNGLPROGRAMBUFFERPARAMETERSIUIVNVPROC) load(userptr, "glProgramBufferParametersIuivNV");
    context->ProgramBufferParametersfvNV = (PFNGLPROGRAMBUFFERPARAMETERSFVNVPROC) load(userptr, "glProgramBufferParametersfvNV");
}
static void glad_gl_load_GL_NV_path_rendering(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_path_rendering) return;
    context->CopyPathNV = (PFNGLCOPYPATHNVPROC) load(userptr, "glCopyPathNV");
    context->CoverFillPathInstancedNV = (PFNGLCOVERFILLPATHINSTANCEDNVPROC) load(userptr, "glCoverFillPathInstancedNV");
    context->CoverFillPathNV = (PFNGLCOVERFILLPATHNVPROC) load(userptr, "glCoverFillPathNV");
    context->CoverStrokePathInstancedNV = (PFNGLCOVERSTROKEPATHINSTANCEDNVPROC) load(userptr, "glCoverStrokePathInstancedNV");
    context->CoverStrokePathNV = (PFNGLCOVERSTROKEPATHNVPROC) load(userptr, "glCoverStrokePathNV");
    context->DeletePathsNV = (PFNGLDELETEPATHSNVPROC) load(userptr, "glDeletePathsNV");
    context->GenPathsNV = (PFNGLGENPATHSNVPROC) load(userptr, "glGenPathsNV");
    context->GetPathColorGenfvNV = (PFNGLGETPATHCOLORGENFVNVPROC) load(userptr, "glGetPathColorGenfvNV");
    context->GetPathColorGenivNV = (PFNGLGETPATHCOLORGENIVNVPROC) load(userptr, "glGetPathColorGenivNV");
    context->GetPathCommandsNV = (PFNGLGETPATHCOMMANDSNVPROC) load(userptr, "glGetPathCommandsNV");
    context->GetPathCoordsNV = (PFNGLGETPATHCOORDSNVPROC) load(userptr, "glGetPathCoordsNV");
    context->GetPathDashArrayNV = (PFNGLGETPATHDASHARRAYNVPROC) load(userptr, "glGetPathDashArrayNV");
    context->GetPathLengthNV = (PFNGLGETPATHLENGTHNVPROC) load(userptr, "glGetPathLengthNV");
    context->GetPathMetricRangeNV = (PFNGLGETPATHMETRICRANGENVPROC) load(userptr, "glGetPathMetricRangeNV");
    context->GetPathMetricsNV = (PFNGLGETPATHMETRICSNVPROC) load(userptr, "glGetPathMetricsNV");
    context->GetPathParameterfvNV = (PFNGLGETPATHPARAMETERFVNVPROC) load(userptr, "glGetPathParameterfvNV");
    context->GetPathParameterivNV = (PFNGLGETPATHPARAMETERIVNVPROC) load(userptr, "glGetPathParameterivNV");
    context->GetPathSpacingNV = (PFNGLGETPATHSPACINGNVPROC) load(userptr, "glGetPathSpacingNV");
    context->GetPathTexGenfvNV = (PFNGLGETPATHTEXGENFVNVPROC) load(userptr, "glGetPathTexGenfvNV");
    context->GetPathTexGenivNV = (PFNGLGETPATHTEXGENIVNVPROC) load(userptr, "glGetPathTexGenivNV");
    context->GetProgramResourcefvNV = (PFNGLGETPROGRAMRESOURCEFVNVPROC) load(userptr, "glGetProgramResourcefvNV");
    context->InterpolatePathsNV = (PFNGLINTERPOLATEPATHSNVPROC) load(userptr, "glInterpolatePathsNV");
    context->IsPathNV = (PFNGLISPATHNVPROC) load(userptr, "glIsPathNV");
    context->IsPointInFillPathNV = (PFNGLISPOINTINFILLPATHNVPROC) load(userptr, "glIsPointInFillPathNV");
    context->IsPointInStrokePathNV = (PFNGLISPOINTINSTROKEPATHNVPROC) load(userptr, "glIsPointInStrokePathNV");
    context->MatrixFrustumEXT = (PFNGLMATRIXFRUSTUMEXTPROC) load(userptr, "glMatrixFrustumEXT");
    context->MatrixLoad3x2fNV = (PFNGLMATRIXLOAD3X2FNVPROC) load(userptr, "glMatrixLoad3x2fNV");
    context->MatrixLoad3x3fNV = (PFNGLMATRIXLOAD3X3FNVPROC) load(userptr, "glMatrixLoad3x3fNV");
    context->MatrixLoadIdentityEXT = (PFNGLMATRIXLOADIDENTITYEXTPROC) load(userptr, "glMatrixLoadIdentityEXT");
    context->MatrixLoadTranspose3x3fNV = (PFNGLMATRIXLOADTRANSPOSE3X3FNVPROC) load(userptr, "glMatrixLoadTranspose3x3fNV");
    context->MatrixLoadTransposedEXT = (PFNGLMATRIXLOADTRANSPOSEDEXTPROC) load(userptr, "glMatrixLoadTransposedEXT");
    context->MatrixLoadTransposefEXT = (PFNGLMATRIXLOADTRANSPOSEFEXTPROC) load(userptr, "glMatrixLoadTransposefEXT");
    context->MatrixLoaddEXT = (PFNGLMATRIXLOADDEXTPROC) load(userptr, "glMatrixLoaddEXT");
    context->MatrixLoadfEXT = (PFNGLMATRIXLOADFEXTPROC) load(userptr, "glMatrixLoadfEXT");
    context->MatrixMult3x2fNV = (PFNGLMATRIXMULT3X2FNVPROC) load(userptr, "glMatrixMult3x2fNV");
    context->MatrixMult3x3fNV = (PFNGLMATRIXMULT3X3FNVPROC) load(userptr, "glMatrixMult3x3fNV");
    context->MatrixMultTranspose3x3fNV = (PFNGLMATRIXMULTTRANSPOSE3X3FNVPROC) load(userptr, "glMatrixMultTranspose3x3fNV");
    context->MatrixMultTransposedEXT = (PFNGLMATRIXMULTTRANSPOSEDEXTPROC) load(userptr, "glMatrixMultTransposedEXT");
    context->MatrixMultTransposefEXT = (PFNGLMATRIXMULTTRANSPOSEFEXTPROC) load(userptr, "glMatrixMultTransposefEXT");
    context->MatrixMultdEXT = (PFNGLMATRIXMULTDEXTPROC) load(userptr, "glMatrixMultdEXT");
    context->MatrixMultfEXT = (PFNGLMATRIXMULTFEXTPROC) load(userptr, "glMatrixMultfEXT");
    context->MatrixOrthoEXT = (PFNGLMATRIXORTHOEXTPROC) load(userptr, "glMatrixOrthoEXT");
    context->MatrixPopEXT = (PFNGLMATRIXPOPEXTPROC) load(userptr, "glMatrixPopEXT");
    context->MatrixPushEXT = (PFNGLMATRIXPUSHEXTPROC) load(userptr, "glMatrixPushEXT");
    context->MatrixRotatedEXT = (PFNGLMATRIXROTATEDEXTPROC) load(userptr, "glMatrixRotatedEXT");
    context->MatrixRotatefEXT = (PFNGLMATRIXROTATEFEXTPROC) load(userptr, "glMatrixRotatefEXT");
    context->MatrixScaledEXT = (PFNGLMATRIXSCALEDEXTPROC) load(userptr, "glMatrixScaledEXT");
    context->MatrixScalefEXT = (PFNGLMATRIXSCALEFEXTPROC) load(userptr, "glMatrixScalefEXT");
    context->MatrixTranslatedEXT = (PFNGLMATRIXTRANSLATEDEXTPROC) load(userptr, "glMatrixTranslatedEXT");
    context->MatrixTranslatefEXT = (PFNGLMATRIXTRANSLATEFEXTPROC) load(userptr, "glMatrixTranslatefEXT");
    context->PathColorGenNV = (PFNGLPATHCOLORGENNVPROC) load(userptr, "glPathColorGenNV");
    context->PathCommandsNV = (PFNGLPATHCOMMANDSNVPROC) load(userptr, "glPathCommandsNV");
    context->PathCoordsNV = (PFNGLPATHCOORDSNVPROC) load(userptr, "glPathCoordsNV");
    context->PathCoverDepthFuncNV = (PFNGLPATHCOVERDEPTHFUNCNVPROC) load(userptr, "glPathCoverDepthFuncNV");
    context->PathDashArrayNV = (PFNGLPATHDASHARRAYNVPROC) load(userptr, "glPathDashArrayNV");
    context->PathFogGenNV = (PFNGLPATHFOGGENNVPROC) load(userptr, "glPathFogGenNV");
    context->PathGlyphIndexArrayNV = (PFNGLPATHGLYPHINDEXARRAYNVPROC) load(userptr, "glPathGlyphIndexArrayNV");
    context->PathGlyphIndexRangeNV = (PFNGLPATHGLYPHINDEXRANGENVPROC) load(userptr, "glPathGlyphIndexRangeNV");
    context->PathGlyphRangeNV = (PFNGLPATHGLYPHRANGENVPROC) load(userptr, "glPathGlyphRangeNV");
    context->PathGlyphsNV = (PFNGLPATHGLYPHSNVPROC) load(userptr, "glPathGlyphsNV");
    context->PathMemoryGlyphIndexArrayNV = (PFNGLPATHMEMORYGLYPHINDEXARRAYNVPROC) load(userptr, "glPathMemoryGlyphIndexArrayNV");
    context->PathParameterfNV = (PFNGLPATHPARAMETERFNVPROC) load(userptr, "glPathParameterfNV");
    context->PathParameterfvNV = (PFNGLPATHPARAMETERFVNVPROC) load(userptr, "glPathParameterfvNV");
    context->PathParameteriNV = (PFNGLPATHPARAMETERINVPROC) load(userptr, "glPathParameteriNV");
    context->PathParameterivNV = (PFNGLPATHPARAMETERIVNVPROC) load(userptr, "glPathParameterivNV");
    context->PathStencilDepthOffsetNV = (PFNGLPATHSTENCILDEPTHOFFSETNVPROC) load(userptr, "glPathStencilDepthOffsetNV");
    context->PathStencilFuncNV = (PFNGLPATHSTENCILFUNCNVPROC) load(userptr, "glPathStencilFuncNV");
    context->PathStringNV = (PFNGLPATHSTRINGNVPROC) load(userptr, "glPathStringNV");
    context->PathSubCommandsNV = (PFNGLPATHSUBCOMMANDSNVPROC) load(userptr, "glPathSubCommandsNV");
    context->PathSubCoordsNV = (PFNGLPATHSUBCOORDSNVPROC) load(userptr, "glPathSubCoordsNV");
    context->PathTexGenNV = (PFNGLPATHTEXGENNVPROC) load(userptr, "glPathTexGenNV");
    context->PointAlongPathNV = (PFNGLPOINTALONGPATHNVPROC) load(userptr, "glPointAlongPathNV");
    context->ProgramPathFragmentInputGenNV = (PFNGLPROGRAMPATHFRAGMENTINPUTGENNVPROC) load(userptr, "glProgramPathFragmentInputGenNV");
    context->StencilFillPathInstancedNV = (PFNGLSTENCILFILLPATHINSTANCEDNVPROC) load(userptr, "glStencilFillPathInstancedNV");
    context->StencilFillPathNV = (PFNGLSTENCILFILLPATHNVPROC) load(userptr, "glStencilFillPathNV");
    context->StencilStrokePathInstancedNV = (PFNGLSTENCILSTROKEPATHINSTANCEDNVPROC) load(userptr, "glStencilStrokePathInstancedNV");
    context->StencilStrokePathNV = (PFNGLSTENCILSTROKEPATHNVPROC) load(userptr, "glStencilStrokePathNV");
    context->StencilThenCoverFillPathInstancedNV = (PFNGLSTENCILTHENCOVERFILLPATHINSTANCEDNVPROC) load(userptr, "glStencilThenCoverFillPathInstancedNV");
    context->StencilThenCoverFillPathNV = (PFNGLSTENCILTHENCOVERFILLPATHNVPROC) load(userptr, "glStencilThenCoverFillPathNV");
    context->StencilThenCoverStrokePathInstancedNV = (PFNGLSTENCILTHENCOVERSTROKEPATHINSTANCEDNVPROC) load(userptr, "glStencilThenCoverStrokePathInstancedNV");
    context->StencilThenCoverStrokePathNV = (PFNGLSTENCILTHENCOVERSTROKEPATHNVPROC) load(userptr, "glStencilThenCoverStrokePathNV");
    context->TransformPathNV = (PFNGLTRANSFORMPATHNVPROC) load(userptr, "glTransformPathNV");
    context->WeightPathsNV = (PFNGLWEIGHTPATHSNVPROC) load(userptr, "glWeightPathsNV");
}
static void glad_gl_load_GL_NV_pixel_data_range(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_pixel_data_range) return;
    context->FlushPixelDataRangeNV = (PFNGLFLUSHPIXELDATARANGENVPROC) load(userptr, "glFlushPixelDataRangeNV");
    context->PixelDataRangeNV = (PFNGLPIXELDATARANGENVPROC) load(userptr, "glPixelDataRangeNV");
}
static void glad_gl_load_GL_NV_point_sprite(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_point_sprite) return;
    context->PointParameteriNV = (PFNGLPOINTPARAMETERINVPROC) load(userptr, "glPointParameteriNV");
    context->PointParameterivNV = (PFNGLPOINTPARAMETERIVNVPROC) load(userptr, "glPointParameterivNV");
}
static void glad_gl_load_GL_NV_present_video(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_present_video) return;
    context->GetVideoi64vNV = (PFNGLGETVIDEOI64VNVPROC) load(userptr, "glGetVideoi64vNV");
    context->GetVideoivNV = (PFNGLGETVIDEOIVNVPROC) load(userptr, "glGetVideoivNV");
    context->GetVideoui64vNV = (PFNGLGETVIDEOUI64VNVPROC) load(userptr, "glGetVideoui64vNV");
    context->GetVideouivNV = (PFNGLGETVIDEOUIVNVPROC) load(userptr, "glGetVideouivNV");
    context->PresentFrameDualFillNV = (PFNGLPRESENTFRAMEDUALFILLNVPROC) load(userptr, "glPresentFrameDualFillNV");
    context->PresentFrameKeyedNV = (PFNGLPRESENTFRAMEKEYEDNVPROC) load(userptr, "glPresentFrameKeyedNV");
}
static void glad_gl_load_GL_NV_primitive_restart(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_primitive_restart) return;
    context->PrimitiveRestartIndexNV = (PFNGLPRIMITIVERESTARTINDEXNVPROC) load(userptr, "glPrimitiveRestartIndexNV");
    context->PrimitiveRestartNV = (PFNGLPRIMITIVERESTARTNVPROC) load(userptr, "glPrimitiveRestartNV");
}
static void glad_gl_load_GL_NV_query_resource(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_query_resource) return;
    context->QueryResourceNV = (PFNGLQUERYRESOURCENVPROC) load(userptr, "glQueryResourceNV");
}
static void glad_gl_load_GL_NV_query_resource_tag(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_query_resource_tag) return;
    context->DeleteQueryResourceTagNV = (PFNGLDELETEQUERYRESOURCETAGNVPROC) load(userptr, "glDeleteQueryResourceTagNV");
    context->GenQueryResourceTagNV = (PFNGLGENQUERYRESOURCETAGNVPROC) load(userptr, "glGenQueryResourceTagNV");
    context->QueryResourceTagNV = (PFNGLQUERYRESOURCETAGNVPROC) load(userptr, "glQueryResourceTagNV");
}
static void glad_gl_load_GL_NV_register_combiners(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_register_combiners) return;
    context->CombinerInputNV = (PFNGLCOMBINERINPUTNVPROC) load(userptr, "glCombinerInputNV");
    context->CombinerOutputNV = (PFNGLCOMBINEROUTPUTNVPROC) load(userptr, "glCombinerOutputNV");
    context->CombinerParameterfNV = (PFNGLCOMBINERPARAMETERFNVPROC) load(userptr, "glCombinerParameterfNV");
    context->CombinerParameterfvNV = (PFNGLCOMBINERPARAMETERFVNVPROC) load(userptr, "glCombinerParameterfvNV");
    context->CombinerParameteriNV = (PFNGLCOMBINERPARAMETERINVPROC) load(userptr, "glCombinerParameteriNV");
    context->CombinerParameterivNV = (PFNGLCOMBINERPARAMETERIVNVPROC) load(userptr, "glCombinerParameterivNV");
    context->FinalCombinerInputNV = (PFNGLFINALCOMBINERINPUTNVPROC) load(userptr, "glFinalCombinerInputNV");
    context->GetCombinerInputParameterfvNV = (PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC) load(userptr, "glGetCombinerInputParameterfvNV");
    context->GetCombinerInputParameterivNV = (PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC) load(userptr, "glGetCombinerInputParameterivNV");
    context->GetCombinerOutputParameterfvNV = (PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC) load(userptr, "glGetCombinerOutputParameterfvNV");
    context->GetCombinerOutputParameterivNV = (PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC) load(userptr, "glGetCombinerOutputParameterivNV");
    context->GetFinalCombinerInputParameterfvNV = (PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC) load(userptr, "glGetFinalCombinerInputParameterfvNV");
    context->GetFinalCombinerInputParameterivNV = (PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC) load(userptr, "glGetFinalCombinerInputParameterivNV");
}
static void glad_gl_load_GL_NV_register_combiners2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_register_combiners2) return;
    context->CombinerStageParameterfvNV = (PFNGLCOMBINERSTAGEPARAMETERFVNVPROC) load(userptr, "glCombinerStageParameterfvNV");
    context->GetCombinerStageParameterfvNV = (PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC) load(userptr, "glGetCombinerStageParameterfvNV");
}
static void glad_gl_load_GL_NV_sample_locations(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_sample_locations) return;
    context->FramebufferSampleLocationsfvNV = (PFNGLFRAMEBUFFERSAMPLELOCATIONSFVNVPROC) load(userptr, "glFramebufferSampleLocationsfvNV");
    context->NamedFramebufferSampleLocationsfvNV = (PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVNVPROC) load(userptr, "glNamedFramebufferSampleLocationsfvNV");
    context->ResolveDepthValuesNV = (PFNGLRESOLVEDEPTHVALUESNVPROC) load(userptr, "glResolveDepthValuesNV");
}
static void glad_gl_load_GL_NV_scissor_exclusive(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_scissor_exclusive) return;
    context->ScissorExclusiveArrayvNV = (PFNGLSCISSOREXCLUSIVEARRAYVNVPROC) load(userptr, "glScissorExclusiveArrayvNV");
    context->ScissorExclusiveNV = (PFNGLSCISSOREXCLUSIVENVPROC) load(userptr, "glScissorExclusiveNV");
}
static void glad_gl_load_GL_NV_shader_buffer_load(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_shader_buffer_load) return;
    context->GetBufferParameterui64vNV = (PFNGLGETBUFFERPARAMETERUI64VNVPROC) load(userptr, "glGetBufferParameterui64vNV");
    context->GetIntegerui64vNV = (PFNGLGETINTEGERUI64VNVPROC) load(userptr, "glGetIntegerui64vNV");
    context->GetNamedBufferParameterui64vNV = (PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC) load(userptr, "glGetNamedBufferParameterui64vNV");
    context->GetUniformui64vNV = (PFNGLGETUNIFORMUI64VNVPROC) load(userptr, "glGetUniformui64vNV");
    context->IsBufferResidentNV = (PFNGLISBUFFERRESIDENTNVPROC) load(userptr, "glIsBufferResidentNV");
    context->IsNamedBufferResidentNV = (PFNGLISNAMEDBUFFERRESIDENTNVPROC) load(userptr, "glIsNamedBufferResidentNV");
    context->MakeBufferNonResidentNV = (PFNGLMAKEBUFFERNONRESIDENTNVPROC) load(userptr, "glMakeBufferNonResidentNV");
    context->MakeBufferResidentNV = (PFNGLMAKEBUFFERRESIDENTNVPROC) load(userptr, "glMakeBufferResidentNV");
    context->MakeNamedBufferNonResidentNV = (PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC) load(userptr, "glMakeNamedBufferNonResidentNV");
    context->MakeNamedBufferResidentNV = (PFNGLMAKENAMEDBUFFERRESIDENTNVPROC) load(userptr, "glMakeNamedBufferResidentNV");
    context->ProgramUniformui64NV = (PFNGLPROGRAMUNIFORMUI64NVPROC) load(userptr, "glProgramUniformui64NV");
    context->ProgramUniformui64vNV = (PFNGLPROGRAMUNIFORMUI64VNVPROC) load(userptr, "glProgramUniformui64vNV");
    context->Uniformui64NV = (PFNGLUNIFORMUI64NVPROC) load(userptr, "glUniformui64NV");
    context->Uniformui64vNV = (PFNGLUNIFORMUI64VNVPROC) load(userptr, "glUniformui64vNV");
}
static void glad_gl_load_GL_NV_shading_rate_image(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_shading_rate_image) return;
    context->BindShadingRateImageNV = (PFNGLBINDSHADINGRATEIMAGENVPROC) load(userptr, "glBindShadingRateImageNV");
    context->GetShadingRateImagePaletteNV = (PFNGLGETSHADINGRATEIMAGEPALETTENVPROC) load(userptr, "glGetShadingRateImagePaletteNV");
    context->GetShadingRateSampleLocationivNV = (PFNGLGETSHADINGRATESAMPLELOCATIONIVNVPROC) load(userptr, "glGetShadingRateSampleLocationivNV");
    context->ShadingRateImageBarrierNV = (PFNGLSHADINGRATEIMAGEBARRIERNVPROC) load(userptr, "glShadingRateImageBarrierNV");
    context->ShadingRateImagePaletteNV = (PFNGLSHADINGRATEIMAGEPALETTENVPROC) load(userptr, "glShadingRateImagePaletteNV");
    context->ShadingRateSampleOrderCustomNV = (PFNGLSHADINGRATESAMPLEORDERCUSTOMNVPROC) load(userptr, "glShadingRateSampleOrderCustomNV");
    context->ShadingRateSampleOrderNV = (PFNGLSHADINGRATESAMPLEORDERNVPROC) load(userptr, "glShadingRateSampleOrderNV");
}
static void glad_gl_load_GL_NV_texture_barrier(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_texture_barrier) return;
    context->TextureBarrierNV = (PFNGLTEXTUREBARRIERNVPROC) load(userptr, "glTextureBarrierNV");
}
static void glad_gl_load_GL_NV_texture_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_texture_multisample) return;
    context->TexImage2DMultisampleCoverageNV = (PFNGLTEXIMAGE2DMULTISAMPLECOVERAGENVPROC) load(userptr, "glTexImage2DMultisampleCoverageNV");
    context->TexImage3DMultisampleCoverageNV = (PFNGLTEXIMAGE3DMULTISAMPLECOVERAGENVPROC) load(userptr, "glTexImage3DMultisampleCoverageNV");
    context->TextureImage2DMultisampleCoverageNV = (PFNGLTEXTUREIMAGE2DMULTISAMPLECOVERAGENVPROC) load(userptr, "glTextureImage2DMultisampleCoverageNV");
    context->TextureImage2DMultisampleNV = (PFNGLTEXTUREIMAGE2DMULTISAMPLENVPROC) load(userptr, "glTextureImage2DMultisampleNV");
    context->TextureImage3DMultisampleCoverageNV = (PFNGLTEXTUREIMAGE3DMULTISAMPLECOVERAGENVPROC) load(userptr, "glTextureImage3DMultisampleCoverageNV");
    context->TextureImage3DMultisampleNV = (PFNGLTEXTUREIMAGE3DMULTISAMPLENVPROC) load(userptr, "glTextureImage3DMultisampleNV");
}
static void glad_gl_load_GL_NV_timeline_semaphore(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_timeline_semaphore) return;
    context->CreateSemaphoresNV = (PFNGLCREATESEMAPHORESNVPROC) load(userptr, "glCreateSemaphoresNV");
    context->GetSemaphoreParameterivNV = (PFNGLGETSEMAPHOREPARAMETERIVNVPROC) load(userptr, "glGetSemaphoreParameterivNV");
    context->SemaphoreParameterivNV = (PFNGLSEMAPHOREPARAMETERIVNVPROC) load(userptr, "glSemaphoreParameterivNV");
}
static void glad_gl_load_GL_NV_transform_feedback(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_transform_feedback) return;
    context->ActiveVaryingNV = (PFNGLACTIVEVARYINGNVPROC) load(userptr, "glActiveVaryingNV");
    context->BeginTransformFeedbackNV = (PFNGLBEGINTRANSFORMFEEDBACKNVPROC) load(userptr, "glBeginTransformFeedbackNV");
    context->BindBufferBaseNV = (PFNGLBINDBUFFERBASENVPROC) load(userptr, "glBindBufferBaseNV");
    context->BindBufferOffsetNV = (PFNGLBINDBUFFEROFFSETNVPROC) load(userptr, "glBindBufferOffsetNV");
    context->BindBufferRangeNV = (PFNGLBINDBUFFERRANGENVPROC) load(userptr, "glBindBufferRangeNV");
    context->EndTransformFeedbackNV = (PFNGLENDTRANSFORMFEEDBACKNVPROC) load(userptr, "glEndTransformFeedbackNV");
    context->GetActiveVaryingNV = (PFNGLGETACTIVEVARYINGNVPROC) load(userptr, "glGetActiveVaryingNV");
    context->GetTransformFeedbackVaryingNV = (PFNGLGETTRANSFORMFEEDBACKVARYINGNVPROC) load(userptr, "glGetTransformFeedbackVaryingNV");
    context->GetVaryingLocationNV = (PFNGLGETVARYINGLOCATIONNVPROC) load(userptr, "glGetVaryingLocationNV");
    context->TransformFeedbackAttribsNV = (PFNGLTRANSFORMFEEDBACKATTRIBSNVPROC) load(userptr, "glTransformFeedbackAttribsNV");
    context->TransformFeedbackStreamAttribsNV = (PFNGLTRANSFORMFEEDBACKSTREAMATTRIBSNVPROC) load(userptr, "glTransformFeedbackStreamAttribsNV");
    context->TransformFeedbackVaryingsNV = (PFNGLTRANSFORMFEEDBACKVARYINGSNVPROC) load(userptr, "glTransformFeedbackVaryingsNV");
}
static void glad_gl_load_GL_NV_transform_feedback2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_transform_feedback2) return;
    context->BindTransformFeedbackNV = (PFNGLBINDTRANSFORMFEEDBACKNVPROC) load(userptr, "glBindTransformFeedbackNV");
    context->DeleteTransformFeedbacksNV = (PFNGLDELETETRANSFORMFEEDBACKSNVPROC) load(userptr, "glDeleteTransformFeedbacksNV");
    context->DrawTransformFeedbackNV = (PFNGLDRAWTRANSFORMFEEDBACKNVPROC) load(userptr, "glDrawTransformFeedbackNV");
    context->GenTransformFeedbacksNV = (PFNGLGENTRANSFORMFEEDBACKSNVPROC) load(userptr, "glGenTransformFeedbacksNV");
    context->IsTransformFeedbackNV = (PFNGLISTRANSFORMFEEDBACKNVPROC) load(userptr, "glIsTransformFeedbackNV");
    context->PauseTransformFeedbackNV = (PFNGLPAUSETRANSFORMFEEDBACKNVPROC) load(userptr, "glPauseTransformFeedbackNV");
    context->ResumeTransformFeedbackNV = (PFNGLRESUMETRANSFORMFEEDBACKNVPROC) load(userptr, "glResumeTransformFeedbackNV");
}
static void glad_gl_load_GL_NV_vdpau_interop(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_vdpau_interop) return;
    context->VDPAUFiniNV = (PFNGLVDPAUFININVPROC) load(userptr, "glVDPAUFiniNV");
    context->VDPAUGetSurfaceivNV = (PFNGLVDPAUGETSURFACEIVNVPROC) load(userptr, "glVDPAUGetSurfaceivNV");
    context->VDPAUInitNV = (PFNGLVDPAUINITNVPROC) load(userptr, "glVDPAUInitNV");
    context->VDPAUIsSurfaceNV = (PFNGLVDPAUISSURFACENVPROC) load(userptr, "glVDPAUIsSurfaceNV");
    context->VDPAUMapSurfacesNV = (PFNGLVDPAUMAPSURFACESNVPROC) load(userptr, "glVDPAUMapSurfacesNV");
    context->VDPAURegisterOutputSurfaceNV = (PFNGLVDPAUREGISTEROUTPUTSURFACENVPROC) load(userptr, "glVDPAURegisterOutputSurfaceNV");
    context->VDPAURegisterVideoSurfaceNV = (PFNGLVDPAUREGISTERVIDEOSURFACENVPROC) load(userptr, "glVDPAURegisterVideoSurfaceNV");
    context->VDPAUSurfaceAccessNV = (PFNGLVDPAUSURFACEACCESSNVPROC) load(userptr, "glVDPAUSurfaceAccessNV");
    context->VDPAUUnmapSurfacesNV = (PFNGLVDPAUUNMAPSURFACESNVPROC) load(userptr, "glVDPAUUnmapSurfacesNV");
    context->VDPAUUnregisterSurfaceNV = (PFNGLVDPAUUNREGISTERSURFACENVPROC) load(userptr, "glVDPAUUnregisterSurfaceNV");
}
static void glad_gl_load_GL_NV_vdpau_interop2(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_vdpau_interop2) return;
    context->VDPAURegisterVideoSurfaceWithPictureStructureNV = (PFNGLVDPAUREGISTERVIDEOSURFACEWITHPICTURESTRUCTURENVPROC) load(userptr, "glVDPAURegisterVideoSurfaceWithPictureStructureNV");
}
static void glad_gl_load_GL_NV_vertex_array_range(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_vertex_array_range) return;
    context->FlushVertexArrayRangeNV = (PFNGLFLUSHVERTEXARRAYRANGENVPROC) load(userptr, "glFlushVertexArrayRangeNV");
    context->VertexArrayRangeNV = (PFNGLVERTEXARRAYRANGENVPROC) load(userptr, "glVertexArrayRangeNV");
}
static void glad_gl_load_GL_NV_vertex_attrib_integer_64bit(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_vertex_attrib_integer_64bit) return;
    context->GetVertexAttribLi64vNV = (PFNGLGETVERTEXATTRIBLI64VNVPROC) load(userptr, "glGetVertexAttribLi64vNV");
    context->GetVertexAttribLui64vNV = (PFNGLGETVERTEXATTRIBLUI64VNVPROC) load(userptr, "glGetVertexAttribLui64vNV");
    context->VertexAttribL1i64NV = (PFNGLVERTEXATTRIBL1I64NVPROC) load(userptr, "glVertexAttribL1i64NV");
    context->VertexAttribL1i64vNV = (PFNGLVERTEXATTRIBL1I64VNVPROC) load(userptr, "glVertexAttribL1i64vNV");
    context->VertexAttribL1ui64NV = (PFNGLVERTEXATTRIBL1UI64NVPROC) load(userptr, "glVertexAttribL1ui64NV");
    context->VertexAttribL1ui64vNV = (PFNGLVERTEXATTRIBL1UI64VNVPROC) load(userptr, "glVertexAttribL1ui64vNV");
    context->VertexAttribL2i64NV = (PFNGLVERTEXATTRIBL2I64NVPROC) load(userptr, "glVertexAttribL2i64NV");
    context->VertexAttribL2i64vNV = (PFNGLVERTEXATTRIBL2I64VNVPROC) load(userptr, "glVertexAttribL2i64vNV");
    context->VertexAttribL2ui64NV = (PFNGLVERTEXATTRIBL2UI64NVPROC) load(userptr, "glVertexAttribL2ui64NV");
    context->VertexAttribL2ui64vNV = (PFNGLVERTEXATTRIBL2UI64VNVPROC) load(userptr, "glVertexAttribL2ui64vNV");
    context->VertexAttribL3i64NV = (PFNGLVERTEXATTRIBL3I64NVPROC) load(userptr, "glVertexAttribL3i64NV");
    context->VertexAttribL3i64vNV = (PFNGLVERTEXATTRIBL3I64VNVPROC) load(userptr, "glVertexAttribL3i64vNV");
    context->VertexAttribL3ui64NV = (PFNGLVERTEXATTRIBL3UI64NVPROC) load(userptr, "glVertexAttribL3ui64NV");
    context->VertexAttribL3ui64vNV = (PFNGLVERTEXATTRIBL3UI64VNVPROC) load(userptr, "glVertexAttribL3ui64vNV");
    context->VertexAttribL4i64NV = (PFNGLVERTEXATTRIBL4I64NVPROC) load(userptr, "glVertexAttribL4i64NV");
    context->VertexAttribL4i64vNV = (PFNGLVERTEXATTRIBL4I64VNVPROC) load(userptr, "glVertexAttribL4i64vNV");
    context->VertexAttribL4ui64NV = (PFNGLVERTEXATTRIBL4UI64NVPROC) load(userptr, "glVertexAttribL4ui64NV");
    context->VertexAttribL4ui64vNV = (PFNGLVERTEXATTRIBL4UI64VNVPROC) load(userptr, "glVertexAttribL4ui64vNV");
    context->VertexAttribLFormatNV = (PFNGLVERTEXATTRIBLFORMATNVPROC) load(userptr, "glVertexAttribLFormatNV");
}
static void glad_gl_load_GL_NV_vertex_buffer_unified_memory(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_vertex_buffer_unified_memory) return;
    context->BufferAddressRangeNV = (PFNGLBUFFERADDRESSRANGENVPROC) load(userptr, "glBufferAddressRangeNV");
    context->ColorFormatNV = (PFNGLCOLORFORMATNVPROC) load(userptr, "glColorFormatNV");
    context->EdgeFlagFormatNV = (PFNGLEDGEFLAGFORMATNVPROC) load(userptr, "glEdgeFlagFormatNV");
    context->FogCoordFormatNV = (PFNGLFOGCOORDFORMATNVPROC) load(userptr, "glFogCoordFormatNV");
    context->GetIntegerui64i_vNV = (PFNGLGETINTEGERUI64I_VNVPROC) load(userptr, "glGetIntegerui64i_vNV");
    context->IndexFormatNV = (PFNGLINDEXFORMATNVPROC) load(userptr, "glIndexFormatNV");
    context->NormalFormatNV = (PFNGLNORMALFORMATNVPROC) load(userptr, "glNormalFormatNV");
    context->SecondaryColorFormatNV = (PFNGLSECONDARYCOLORFORMATNVPROC) load(userptr, "glSecondaryColorFormatNV");
    context->TexCoordFormatNV = (PFNGLTEXCOORDFORMATNVPROC) load(userptr, "glTexCoordFormatNV");
    context->VertexAttribFormatNV = (PFNGLVERTEXATTRIBFORMATNVPROC) load(userptr, "glVertexAttribFormatNV");
    context->VertexAttribIFormatNV = (PFNGLVERTEXATTRIBIFORMATNVPROC) load(userptr, "glVertexAttribIFormatNV");
    context->VertexFormatNV = (PFNGLVERTEXFORMATNVPROC) load(userptr, "glVertexFormatNV");
}
static void glad_gl_load_GL_NV_vertex_program(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_vertex_program) return;
    context->AreProgramsResidentNV = (PFNGLAREPROGRAMSRESIDENTNVPROC) load(userptr, "glAreProgramsResidentNV");
    context->BindProgramNV = (PFNGLBINDPROGRAMNVPROC) load(userptr, "glBindProgramNV");
    context->DeleteProgramsNV = (PFNGLDELETEPROGRAMSNVPROC) load(userptr, "glDeleteProgramsNV");
    context->ExecuteProgramNV = (PFNGLEXECUTEPROGRAMNVPROC) load(userptr, "glExecuteProgramNV");
    context->GenProgramsNV = (PFNGLGENPROGRAMSNVPROC) load(userptr, "glGenProgramsNV");
    context->GetProgramParameterdvNV = (PFNGLGETPROGRAMPARAMETERDVNVPROC) load(userptr, "glGetProgramParameterdvNV");
    context->GetProgramParameterfvNV = (PFNGLGETPROGRAMPARAMETERFVNVPROC) load(userptr, "glGetProgramParameterfvNV");
    context->GetProgramStringNV = (PFNGLGETPROGRAMSTRINGNVPROC) load(userptr, "glGetProgramStringNV");
    context->GetProgramivNV = (PFNGLGETPROGRAMIVNVPROC) load(userptr, "glGetProgramivNV");
    context->GetTrackMatrixivNV = (PFNGLGETTRACKMATRIXIVNVPROC) load(userptr, "glGetTrackMatrixivNV");
    context->GetVertexAttribPointervNV = (PFNGLGETVERTEXATTRIBPOINTERVNVPROC) load(userptr, "glGetVertexAttribPointervNV");
    context->GetVertexAttribdvNV = (PFNGLGETVERTEXATTRIBDVNVPROC) load(userptr, "glGetVertexAttribdvNV");
    context->GetVertexAttribfvNV = (PFNGLGETVERTEXATTRIBFVNVPROC) load(userptr, "glGetVertexAttribfvNV");
    context->GetVertexAttribivNV = (PFNGLGETVERTEXATTRIBIVNVPROC) load(userptr, "glGetVertexAttribivNV");
    context->IsProgramNV = (PFNGLISPROGRAMNVPROC) load(userptr, "glIsProgramNV");
    context->LoadProgramNV = (PFNGLLOADPROGRAMNVPROC) load(userptr, "glLoadProgramNV");
    context->ProgramParameter4dNV = (PFNGLPROGRAMPARAMETER4DNVPROC) load(userptr, "glProgramParameter4dNV");
    context->ProgramParameter4dvNV = (PFNGLPROGRAMPARAMETER4DVNVPROC) load(userptr, "glProgramParameter4dvNV");
    context->ProgramParameter4fNV = (PFNGLPROGRAMPARAMETER4FNVPROC) load(userptr, "glProgramParameter4fNV");
    context->ProgramParameter4fvNV = (PFNGLPROGRAMPARAMETER4FVNVPROC) load(userptr, "glProgramParameter4fvNV");
    context->ProgramParameters4dvNV = (PFNGLPROGRAMPARAMETERS4DVNVPROC) load(userptr, "glProgramParameters4dvNV");
    context->ProgramParameters4fvNV = (PFNGLPROGRAMPARAMETERS4FVNVPROC) load(userptr, "glProgramParameters4fvNV");
    context->RequestResidentProgramsNV = (PFNGLREQUESTRESIDENTPROGRAMSNVPROC) load(userptr, "glRequestResidentProgramsNV");
    context->TrackMatrixNV = (PFNGLTRACKMATRIXNVPROC) load(userptr, "glTrackMatrixNV");
    context->VertexAttrib1dNV = (PFNGLVERTEXATTRIB1DNVPROC) load(userptr, "glVertexAttrib1dNV");
    context->VertexAttrib1dvNV = (PFNGLVERTEXATTRIB1DVNVPROC) load(userptr, "glVertexAttrib1dvNV");
    context->VertexAttrib1fNV = (PFNGLVERTEXATTRIB1FNVPROC) load(userptr, "glVertexAttrib1fNV");
    context->VertexAttrib1fvNV = (PFNGLVERTEXATTRIB1FVNVPROC) load(userptr, "glVertexAttrib1fvNV");
    context->VertexAttrib1sNV = (PFNGLVERTEXATTRIB1SNVPROC) load(userptr, "glVertexAttrib1sNV");
    context->VertexAttrib1svNV = (PFNGLVERTEXATTRIB1SVNVPROC) load(userptr, "glVertexAttrib1svNV");
    context->VertexAttrib2dNV = (PFNGLVERTEXATTRIB2DNVPROC) load(userptr, "glVertexAttrib2dNV");
    context->VertexAttrib2dvNV = (PFNGLVERTEXATTRIB2DVNVPROC) load(userptr, "glVertexAttrib2dvNV");
    context->VertexAttrib2fNV = (PFNGLVERTEXATTRIB2FNVPROC) load(userptr, "glVertexAttrib2fNV");
    context->VertexAttrib2fvNV = (PFNGLVERTEXATTRIB2FVNVPROC) load(userptr, "glVertexAttrib2fvNV");
    context->VertexAttrib2sNV = (PFNGLVERTEXATTRIB2SNVPROC) load(userptr, "glVertexAttrib2sNV");
    context->VertexAttrib2svNV = (PFNGLVERTEXATTRIB2SVNVPROC) load(userptr, "glVertexAttrib2svNV");
    context->VertexAttrib3dNV = (PFNGLVERTEXATTRIB3DNVPROC) load(userptr, "glVertexAttrib3dNV");
    context->VertexAttrib3dvNV = (PFNGLVERTEXATTRIB3DVNVPROC) load(userptr, "glVertexAttrib3dvNV");
    context->VertexAttrib3fNV = (PFNGLVERTEXATTRIB3FNVPROC) load(userptr, "glVertexAttrib3fNV");
    context->VertexAttrib3fvNV = (PFNGLVERTEXATTRIB3FVNVPROC) load(userptr, "glVertexAttrib3fvNV");
    context->VertexAttrib3sNV = (PFNGLVERTEXATTRIB3SNVPROC) load(userptr, "glVertexAttrib3sNV");
    context->VertexAttrib3svNV = (PFNGLVERTEXATTRIB3SVNVPROC) load(userptr, "glVertexAttrib3svNV");
    context->VertexAttrib4dNV = (PFNGLVERTEXATTRIB4DNVPROC) load(userptr, "glVertexAttrib4dNV");
    context->VertexAttrib4dvNV = (PFNGLVERTEXATTRIB4DVNVPROC) load(userptr, "glVertexAttrib4dvNV");
    context->VertexAttrib4fNV = (PFNGLVERTEXATTRIB4FNVPROC) load(userptr, "glVertexAttrib4fNV");
    context->VertexAttrib4fvNV = (PFNGLVERTEXATTRIB4FVNVPROC) load(userptr, "glVertexAttrib4fvNV");
    context->VertexAttrib4sNV = (PFNGLVERTEXATTRIB4SNVPROC) load(userptr, "glVertexAttrib4sNV");
    context->VertexAttrib4svNV = (PFNGLVERTEXATTRIB4SVNVPROC) load(userptr, "glVertexAttrib4svNV");
    context->VertexAttrib4ubNV = (PFNGLVERTEXATTRIB4UBNVPROC) load(userptr, "glVertexAttrib4ubNV");
    context->VertexAttrib4ubvNV = (PFNGLVERTEXATTRIB4UBVNVPROC) load(userptr, "glVertexAttrib4ubvNV");
    context->VertexAttribPointerNV = (PFNGLVERTEXATTRIBPOINTERNVPROC) load(userptr, "glVertexAttribPointerNV");
    context->VertexAttribs1dvNV = (PFNGLVERTEXATTRIBS1DVNVPROC) load(userptr, "glVertexAttribs1dvNV");
    context->VertexAttribs1fvNV = (PFNGLVERTEXATTRIBS1FVNVPROC) load(userptr, "glVertexAttribs1fvNV");
    context->VertexAttribs1svNV = (PFNGLVERTEXATTRIBS1SVNVPROC) load(userptr, "glVertexAttribs1svNV");
    context->VertexAttribs2dvNV = (PFNGLVERTEXATTRIBS2DVNVPROC) load(userptr, "glVertexAttribs2dvNV");
    context->VertexAttribs2fvNV = (PFNGLVERTEXATTRIBS2FVNVPROC) load(userptr, "glVertexAttribs2fvNV");
    context->VertexAttribs2svNV = (PFNGLVERTEXATTRIBS2SVNVPROC) load(userptr, "glVertexAttribs2svNV");
    context->VertexAttribs3dvNV = (PFNGLVERTEXATTRIBS3DVNVPROC) load(userptr, "glVertexAttribs3dvNV");
    context->VertexAttribs3fvNV = (PFNGLVERTEXATTRIBS3FVNVPROC) load(userptr, "glVertexAttribs3fvNV");
    context->VertexAttribs3svNV = (PFNGLVERTEXATTRIBS3SVNVPROC) load(userptr, "glVertexAttribs3svNV");
    context->VertexAttribs4dvNV = (PFNGLVERTEXATTRIBS4DVNVPROC) load(userptr, "glVertexAttribs4dvNV");
    context->VertexAttribs4fvNV = (PFNGLVERTEXATTRIBS4FVNVPROC) load(userptr, "glVertexAttribs4fvNV");
    context->VertexAttribs4svNV = (PFNGLVERTEXATTRIBS4SVNVPROC) load(userptr, "glVertexAttribs4svNV");
    context->VertexAttribs4ubvNV = (PFNGLVERTEXATTRIBS4UBVNVPROC) load(userptr, "glVertexAttribs4ubvNV");
}
static void glad_gl_load_GL_NV_vertex_program4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_vertex_program4) return;
    context->GetVertexAttribIivEXT = (PFNGLGETVERTEXATTRIBIIVEXTPROC) load(userptr, "glGetVertexAttribIivEXT");
    context->GetVertexAttribIuivEXT = (PFNGLGETVERTEXATTRIBIUIVEXTPROC) load(userptr, "glGetVertexAttribIuivEXT");
    context->VertexAttribI1iEXT = (PFNGLVERTEXATTRIBI1IEXTPROC) load(userptr, "glVertexAttribI1iEXT");
    context->VertexAttribI1ivEXT = (PFNGLVERTEXATTRIBI1IVEXTPROC) load(userptr, "glVertexAttribI1ivEXT");
    context->VertexAttribI1uiEXT = (PFNGLVERTEXATTRIBI1UIEXTPROC) load(userptr, "glVertexAttribI1uiEXT");
    context->VertexAttribI1uivEXT = (PFNGLVERTEXATTRIBI1UIVEXTPROC) load(userptr, "glVertexAttribI1uivEXT");
    context->VertexAttribI2iEXT = (PFNGLVERTEXATTRIBI2IEXTPROC) load(userptr, "glVertexAttribI2iEXT");
    context->VertexAttribI2ivEXT = (PFNGLVERTEXATTRIBI2IVEXTPROC) load(userptr, "glVertexAttribI2ivEXT");
    context->VertexAttribI2uiEXT = (PFNGLVERTEXATTRIBI2UIEXTPROC) load(userptr, "glVertexAttribI2uiEXT");
    context->VertexAttribI2uivEXT = (PFNGLVERTEXATTRIBI2UIVEXTPROC) load(userptr, "glVertexAttribI2uivEXT");
    context->VertexAttribI3iEXT = (PFNGLVERTEXATTRIBI3IEXTPROC) load(userptr, "glVertexAttribI3iEXT");
    context->VertexAttribI3ivEXT = (PFNGLVERTEXATTRIBI3IVEXTPROC) load(userptr, "glVertexAttribI3ivEXT");
    context->VertexAttribI3uiEXT = (PFNGLVERTEXATTRIBI3UIEXTPROC) load(userptr, "glVertexAttribI3uiEXT");
    context->VertexAttribI3uivEXT = (PFNGLVERTEXATTRIBI3UIVEXTPROC) load(userptr, "glVertexAttribI3uivEXT");
    context->VertexAttribI4bvEXT = (PFNGLVERTEXATTRIBI4BVEXTPROC) load(userptr, "glVertexAttribI4bvEXT");
    context->VertexAttribI4iEXT = (PFNGLVERTEXATTRIBI4IEXTPROC) load(userptr, "glVertexAttribI4iEXT");
    context->VertexAttribI4ivEXT = (PFNGLVERTEXATTRIBI4IVEXTPROC) load(userptr, "glVertexAttribI4ivEXT");
    context->VertexAttribI4svEXT = (PFNGLVERTEXATTRIBI4SVEXTPROC) load(userptr, "glVertexAttribI4svEXT");
    context->VertexAttribI4ubvEXT = (PFNGLVERTEXATTRIBI4UBVEXTPROC) load(userptr, "glVertexAttribI4ubvEXT");
    context->VertexAttribI4uiEXT = (PFNGLVERTEXATTRIBI4UIEXTPROC) load(userptr, "glVertexAttribI4uiEXT");
    context->VertexAttribI4uivEXT = (PFNGLVERTEXATTRIBI4UIVEXTPROC) load(userptr, "glVertexAttribI4uivEXT");
    context->VertexAttribI4usvEXT = (PFNGLVERTEXATTRIBI4USVEXTPROC) load(userptr, "glVertexAttribI4usvEXT");
    context->VertexAttribIPointerEXT = (PFNGLVERTEXATTRIBIPOINTEREXTPROC) load(userptr, "glVertexAttribIPointerEXT");
}
static void glad_gl_load_GL_NV_video_capture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_video_capture) return;
    context->BeginVideoCaptureNV = (PFNGLBEGINVIDEOCAPTURENVPROC) load(userptr, "glBeginVideoCaptureNV");
    context->BindVideoCaptureStreamBufferNV = (PFNGLBINDVIDEOCAPTURESTREAMBUFFERNVPROC) load(userptr, "glBindVideoCaptureStreamBufferNV");
    context->BindVideoCaptureStreamTextureNV = (PFNGLBINDVIDEOCAPTURESTREAMTEXTURENVPROC) load(userptr, "glBindVideoCaptureStreamTextureNV");
    context->EndVideoCaptureNV = (PFNGLENDVIDEOCAPTURENVPROC) load(userptr, "glEndVideoCaptureNV");
    context->GetVideoCaptureStreamdvNV = (PFNGLGETVIDEOCAPTURESTREAMDVNVPROC) load(userptr, "glGetVideoCaptureStreamdvNV");
    context->GetVideoCaptureStreamfvNV = (PFNGLGETVIDEOCAPTURESTREAMFVNVPROC) load(userptr, "glGetVideoCaptureStreamfvNV");
    context->GetVideoCaptureStreamivNV = (PFNGLGETVIDEOCAPTURESTREAMIVNVPROC) load(userptr, "glGetVideoCaptureStreamivNV");
    context->GetVideoCaptureivNV = (PFNGLGETVIDEOCAPTUREIVNVPROC) load(userptr, "glGetVideoCaptureivNV");
    context->VideoCaptureNV = (PFNGLVIDEOCAPTURENVPROC) load(userptr, "glVideoCaptureNV");
    context->VideoCaptureStreamParameterdvNV = (PFNGLVIDEOCAPTURESTREAMPARAMETERDVNVPROC) load(userptr, "glVideoCaptureStreamParameterdvNV");
    context->VideoCaptureStreamParameterfvNV = (PFNGLVIDEOCAPTURESTREAMPARAMETERFVNVPROC) load(userptr, "glVideoCaptureStreamParameterfvNV");
    context->VideoCaptureStreamParameterivNV = (PFNGLVIDEOCAPTURESTREAMPARAMETERIVNVPROC) load(userptr, "glVideoCaptureStreamParameterivNV");
}
static void glad_gl_load_GL_NV_viewport_swizzle(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->NV_viewport_swizzle) return;
    context->ViewportSwizzleNV = (PFNGLVIEWPORTSWIZZLENVPROC) load(userptr, "glViewportSwizzleNV");
}
static void glad_gl_load_GL_OES_byte_coordinates(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->OES_byte_coordinates) return;
    context->MultiTexCoord1bOES = (PFNGLMULTITEXCOORD1BOESPROC) load(userptr, "glMultiTexCoord1bOES");
    context->MultiTexCoord1bvOES = (PFNGLMULTITEXCOORD1BVOESPROC) load(userptr, "glMultiTexCoord1bvOES");
    context->MultiTexCoord2bOES = (PFNGLMULTITEXCOORD2BOESPROC) load(userptr, "glMultiTexCoord2bOES");
    context->MultiTexCoord2bvOES = (PFNGLMULTITEXCOORD2BVOESPROC) load(userptr, "glMultiTexCoord2bvOES");
    context->MultiTexCoord3bOES = (PFNGLMULTITEXCOORD3BOESPROC) load(userptr, "glMultiTexCoord3bOES");
    context->MultiTexCoord3bvOES = (PFNGLMULTITEXCOORD3BVOESPROC) load(userptr, "glMultiTexCoord3bvOES");
    context->MultiTexCoord4bOES = (PFNGLMULTITEXCOORD4BOESPROC) load(userptr, "glMultiTexCoord4bOES");
    context->MultiTexCoord4bvOES = (PFNGLMULTITEXCOORD4BVOESPROC) load(userptr, "glMultiTexCoord4bvOES");
    context->TexCoord1bOES = (PFNGLTEXCOORD1BOESPROC) load(userptr, "glTexCoord1bOES");
    context->TexCoord1bvOES = (PFNGLTEXCOORD1BVOESPROC) load(userptr, "glTexCoord1bvOES");
    context->TexCoord2bOES = (PFNGLTEXCOORD2BOESPROC) load(userptr, "glTexCoord2bOES");
    context->TexCoord2bvOES = (PFNGLTEXCOORD2BVOESPROC) load(userptr, "glTexCoord2bvOES");
    context->TexCoord3bOES = (PFNGLTEXCOORD3BOESPROC) load(userptr, "glTexCoord3bOES");
    context->TexCoord3bvOES = (PFNGLTEXCOORD3BVOESPROC) load(userptr, "glTexCoord3bvOES");
    context->TexCoord4bOES = (PFNGLTEXCOORD4BOESPROC) load(userptr, "glTexCoord4bOES");
    context->TexCoord4bvOES = (PFNGLTEXCOORD4BVOESPROC) load(userptr, "glTexCoord4bvOES");
    context->Vertex2bOES = (PFNGLVERTEX2BOESPROC) load(userptr, "glVertex2bOES");
    context->Vertex2bvOES = (PFNGLVERTEX2BVOESPROC) load(userptr, "glVertex2bvOES");
    context->Vertex3bOES = (PFNGLVERTEX3BOESPROC) load(userptr, "glVertex3bOES");
    context->Vertex3bvOES = (PFNGLVERTEX3BVOESPROC) load(userptr, "glVertex3bvOES");
    context->Vertex4bOES = (PFNGLVERTEX4BOESPROC) load(userptr, "glVertex4bOES");
    context->Vertex4bvOES = (PFNGLVERTEX4BVOESPROC) load(userptr, "glVertex4bvOES");
}
static void glad_gl_load_GL_OES_fixed_point(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->OES_fixed_point) return;
    context->AccumxOES = (PFNGLACCUMXOESPROC) load(userptr, "glAccumxOES");
    context->AlphaFuncxOES = (PFNGLALPHAFUNCXOESPROC) load(userptr, "glAlphaFuncxOES");
    context->BitmapxOES = (PFNGLBITMAPXOESPROC) load(userptr, "glBitmapxOES");
    context->BlendColorxOES = (PFNGLBLENDCOLORXOESPROC) load(userptr, "glBlendColorxOES");
    context->ClearAccumxOES = (PFNGLCLEARACCUMXOESPROC) load(userptr, "glClearAccumxOES");
    context->ClearColorxOES = (PFNGLCLEARCOLORXOESPROC) load(userptr, "glClearColorxOES");
    context->ClearDepthxOES = (PFNGLCLEARDEPTHXOESPROC) load(userptr, "glClearDepthxOES");
    context->ClipPlanexOES = (PFNGLCLIPPLANEXOESPROC) load(userptr, "glClipPlanexOES");
    context->Color3xOES = (PFNGLCOLOR3XOESPROC) load(userptr, "glColor3xOES");
    context->Color3xvOES = (PFNGLCOLOR3XVOESPROC) load(userptr, "glColor3xvOES");
    context->Color4xOES = (PFNGLCOLOR4XOESPROC) load(userptr, "glColor4xOES");
    context->Color4xvOES = (PFNGLCOLOR4XVOESPROC) load(userptr, "glColor4xvOES");
    context->ConvolutionParameterxOES = (PFNGLCONVOLUTIONPARAMETERXOESPROC) load(userptr, "glConvolutionParameterxOES");
    context->ConvolutionParameterxvOES = (PFNGLCONVOLUTIONPARAMETERXVOESPROC) load(userptr, "glConvolutionParameterxvOES");
    context->DepthRangexOES = (PFNGLDEPTHRANGEXOESPROC) load(userptr, "glDepthRangexOES");
    context->EvalCoord1xOES = (PFNGLEVALCOORD1XOESPROC) load(userptr, "glEvalCoord1xOES");
    context->EvalCoord1xvOES = (PFNGLEVALCOORD1XVOESPROC) load(userptr, "glEvalCoord1xvOES");
    context->EvalCoord2xOES = (PFNGLEVALCOORD2XOESPROC) load(userptr, "glEvalCoord2xOES");
    context->EvalCoord2xvOES = (PFNGLEVALCOORD2XVOESPROC) load(userptr, "glEvalCoord2xvOES");
    context->FeedbackBufferxOES = (PFNGLFEEDBACKBUFFERXOESPROC) load(userptr, "glFeedbackBufferxOES");
    context->FogxOES = (PFNGLFOGXOESPROC) load(userptr, "glFogxOES");
    context->FogxvOES = (PFNGLFOGXVOESPROC) load(userptr, "glFogxvOES");
    context->FrustumxOES = (PFNGLFRUSTUMXOESPROC) load(userptr, "glFrustumxOES");
    context->GetClipPlanexOES = (PFNGLGETCLIPPLANEXOESPROC) load(userptr, "glGetClipPlanexOES");
    context->GetConvolutionParameterxvOES = (PFNGLGETCONVOLUTIONPARAMETERXVOESPROC) load(userptr, "glGetConvolutionParameterxvOES");
    context->GetFixedvOES = (PFNGLGETFIXEDVOESPROC) load(userptr, "glGetFixedvOES");
    context->GetHistogramParameterxvOES = (PFNGLGETHISTOGRAMPARAMETERXVOESPROC) load(userptr, "glGetHistogramParameterxvOES");
    context->GetLightxOES = (PFNGLGETLIGHTXOESPROC) load(userptr, "glGetLightxOES");
    context->GetMapxvOES = (PFNGLGETMAPXVOESPROC) load(userptr, "glGetMapxvOES");
    context->GetMaterialxOES = (PFNGLGETMATERIALXOESPROC) load(userptr, "glGetMaterialxOES");
    context->GetPixelMapxv = (PFNGLGETPIXELMAPXVPROC) load(userptr, "glGetPixelMapxv");
    context->GetTexEnvxvOES = (PFNGLGETTEXENVXVOESPROC) load(userptr, "glGetTexEnvxvOES");
    context->GetTexGenxvOES = (PFNGLGETTEXGENXVOESPROC) load(userptr, "glGetTexGenxvOES");
    context->GetTexLevelParameterxvOES = (PFNGLGETTEXLEVELPARAMETERXVOESPROC) load(userptr, "glGetTexLevelParameterxvOES");
    context->GetTexParameterxvOES = (PFNGLGETTEXPARAMETERXVOESPROC) load(userptr, "glGetTexParameterxvOES");
    context->IndexxOES = (PFNGLINDEXXOESPROC) load(userptr, "glIndexxOES");
    context->IndexxvOES = (PFNGLINDEXXVOESPROC) load(userptr, "glIndexxvOES");
    context->LightModelxOES = (PFNGLLIGHTMODELXOESPROC) load(userptr, "glLightModelxOES");
    context->LightModelxvOES = (PFNGLLIGHTMODELXVOESPROC) load(userptr, "glLightModelxvOES");
    context->LightxOES = (PFNGLLIGHTXOESPROC) load(userptr, "glLightxOES");
    context->LightxvOES = (PFNGLLIGHTXVOESPROC) load(userptr, "glLightxvOES");
    context->LineWidthxOES = (PFNGLLINEWIDTHXOESPROC) load(userptr, "glLineWidthxOES");
    context->LoadMatrixxOES = (PFNGLLOADMATRIXXOESPROC) load(userptr, "glLoadMatrixxOES");
    context->LoadTransposeMatrixxOES = (PFNGLLOADTRANSPOSEMATRIXXOESPROC) load(userptr, "glLoadTransposeMatrixxOES");
    context->Map1xOES = (PFNGLMAP1XOESPROC) load(userptr, "glMap1xOES");
    context->Map2xOES = (PFNGLMAP2XOESPROC) load(userptr, "glMap2xOES");
    context->MapGrid1xOES = (PFNGLMAPGRID1XOESPROC) load(userptr, "glMapGrid1xOES");
    context->MapGrid2xOES = (PFNGLMAPGRID2XOESPROC) load(userptr, "glMapGrid2xOES");
    context->MaterialxOES = (PFNGLMATERIALXOESPROC) load(userptr, "glMaterialxOES");
    context->MaterialxvOES = (PFNGLMATERIALXVOESPROC) load(userptr, "glMaterialxvOES");
    context->MultMatrixxOES = (PFNGLMULTMATRIXXOESPROC) load(userptr, "glMultMatrixxOES");
    context->MultTransposeMatrixxOES = (PFNGLMULTTRANSPOSEMATRIXXOESPROC) load(userptr, "glMultTransposeMatrixxOES");
    context->MultiTexCoord1xOES = (PFNGLMULTITEXCOORD1XOESPROC) load(userptr, "glMultiTexCoord1xOES");
    context->MultiTexCoord1xvOES = (PFNGLMULTITEXCOORD1XVOESPROC) load(userptr, "glMultiTexCoord1xvOES");
    context->MultiTexCoord2xOES = (PFNGLMULTITEXCOORD2XOESPROC) load(userptr, "glMultiTexCoord2xOES");
    context->MultiTexCoord2xvOES = (PFNGLMULTITEXCOORD2XVOESPROC) load(userptr, "glMultiTexCoord2xvOES");
    context->MultiTexCoord3xOES = (PFNGLMULTITEXCOORD3XOESPROC) load(userptr, "glMultiTexCoord3xOES");
    context->MultiTexCoord3xvOES = (PFNGLMULTITEXCOORD3XVOESPROC) load(userptr, "glMultiTexCoord3xvOES");
    context->MultiTexCoord4xOES = (PFNGLMULTITEXCOORD4XOESPROC) load(userptr, "glMultiTexCoord4xOES");
    context->MultiTexCoord4xvOES = (PFNGLMULTITEXCOORD4XVOESPROC) load(userptr, "glMultiTexCoord4xvOES");
    context->Normal3xOES = (PFNGLNORMAL3XOESPROC) load(userptr, "glNormal3xOES");
    context->Normal3xvOES = (PFNGLNORMAL3XVOESPROC) load(userptr, "glNormal3xvOES");
    context->OrthoxOES = (PFNGLORTHOXOESPROC) load(userptr, "glOrthoxOES");
    context->PassThroughxOES = (PFNGLPASSTHROUGHXOESPROC) load(userptr, "glPassThroughxOES");
    context->PixelMapx = (PFNGLPIXELMAPXPROC) load(userptr, "glPixelMapx");
    context->PixelStorex = (PFNGLPIXELSTOREXPROC) load(userptr, "glPixelStorex");
    context->PixelTransferxOES = (PFNGLPIXELTRANSFERXOESPROC) load(userptr, "glPixelTransferxOES");
    context->PixelZoomxOES = (PFNGLPIXELZOOMXOESPROC) load(userptr, "glPixelZoomxOES");
    context->PointParameterxvOES = (PFNGLPOINTPARAMETERXVOESPROC) load(userptr, "glPointParameterxvOES");
    context->PointSizexOES = (PFNGLPOINTSIZEXOESPROC) load(userptr, "glPointSizexOES");
    context->PolygonOffsetxOES = (PFNGLPOLYGONOFFSETXOESPROC) load(userptr, "glPolygonOffsetxOES");
    context->PrioritizeTexturesxOES = (PFNGLPRIORITIZETEXTURESXOESPROC) load(userptr, "glPrioritizeTexturesxOES");
    context->RasterPos2xOES = (PFNGLRASTERPOS2XOESPROC) load(userptr, "glRasterPos2xOES");
    context->RasterPos2xvOES = (PFNGLRASTERPOS2XVOESPROC) load(userptr, "glRasterPos2xvOES");
    context->RasterPos3xOES = (PFNGLRASTERPOS3XOESPROC) load(userptr, "glRasterPos3xOES");
    context->RasterPos3xvOES = (PFNGLRASTERPOS3XVOESPROC) load(userptr, "glRasterPos3xvOES");
    context->RasterPos4xOES = (PFNGLRASTERPOS4XOESPROC) load(userptr, "glRasterPos4xOES");
    context->RasterPos4xvOES = (PFNGLRASTERPOS4XVOESPROC) load(userptr, "glRasterPos4xvOES");
    context->RectxOES = (PFNGLRECTXOESPROC) load(userptr, "glRectxOES");
    context->RectxvOES = (PFNGLRECTXVOESPROC) load(userptr, "glRectxvOES");
    context->RotatexOES = (PFNGLROTATEXOESPROC) load(userptr, "glRotatexOES");
    context->ScalexOES = (PFNGLSCALEXOESPROC) load(userptr, "glScalexOES");
    context->TexCoord1xOES = (PFNGLTEXCOORD1XOESPROC) load(userptr, "glTexCoord1xOES");
    context->TexCoord1xvOES = (PFNGLTEXCOORD1XVOESPROC) load(userptr, "glTexCoord1xvOES");
    context->TexCoord2xOES = (PFNGLTEXCOORD2XOESPROC) load(userptr, "glTexCoord2xOES");
    context->TexCoord2xvOES = (PFNGLTEXCOORD2XVOESPROC) load(userptr, "glTexCoord2xvOES");
    context->TexCoord3xOES = (PFNGLTEXCOORD3XOESPROC) load(userptr, "glTexCoord3xOES");
    context->TexCoord3xvOES = (PFNGLTEXCOORD3XVOESPROC) load(userptr, "glTexCoord3xvOES");
    context->TexCoord4xOES = (PFNGLTEXCOORD4XOESPROC) load(userptr, "glTexCoord4xOES");
    context->TexCoord4xvOES = (PFNGLTEXCOORD4XVOESPROC) load(userptr, "glTexCoord4xvOES");
    context->TexEnvxOES = (PFNGLTEXENVXOESPROC) load(userptr, "glTexEnvxOES");
    context->TexEnvxvOES = (PFNGLTEXENVXVOESPROC) load(userptr, "glTexEnvxvOES");
    context->TexGenxOES = (PFNGLTEXGENXOESPROC) load(userptr, "glTexGenxOES");
    context->TexGenxvOES = (PFNGLTEXGENXVOESPROC) load(userptr, "glTexGenxvOES");
    context->TexParameterxOES = (PFNGLTEXPARAMETERXOESPROC) load(userptr, "glTexParameterxOES");
    context->TexParameterxvOES = (PFNGLTEXPARAMETERXVOESPROC) load(userptr, "glTexParameterxvOES");
    context->TranslatexOES = (PFNGLTRANSLATEXOESPROC) load(userptr, "glTranslatexOES");
    context->Vertex2xOES = (PFNGLVERTEX2XOESPROC) load(userptr, "glVertex2xOES");
    context->Vertex2xvOES = (PFNGLVERTEX2XVOESPROC) load(userptr, "glVertex2xvOES");
    context->Vertex3xOES = (PFNGLVERTEX3XOESPROC) load(userptr, "glVertex3xOES");
    context->Vertex3xvOES = (PFNGLVERTEX3XVOESPROC) load(userptr, "glVertex3xvOES");
    context->Vertex4xOES = (PFNGLVERTEX4XOESPROC) load(userptr, "glVertex4xOES");
    context->Vertex4xvOES = (PFNGLVERTEX4XVOESPROC) load(userptr, "glVertex4xvOES");
}
static void glad_gl_load_GL_OES_query_matrix(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->OES_query_matrix) return;
    context->QueryMatrixxOES = (PFNGLQUERYMATRIXXOESPROC) load(userptr, "glQueryMatrixxOES");
}
static void glad_gl_load_GL_OES_single_precision(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->OES_single_precision) return;
    context->ClearDepthfOES = (PFNGLCLEARDEPTHFOESPROC) load(userptr, "glClearDepthfOES");
    context->ClipPlanefOES = (PFNGLCLIPPLANEFOESPROC) load(userptr, "glClipPlanefOES");
    context->DepthRangefOES = (PFNGLDEPTHRANGEFOESPROC) load(userptr, "glDepthRangefOES");
    context->FrustumfOES = (PFNGLFRUSTUMFOESPROC) load(userptr, "glFrustumfOES");
    context->GetClipPlanefOES = (PFNGLGETCLIPPLANEFOESPROC) load(userptr, "glGetClipPlanefOES");
    context->OrthofOES = (PFNGLORTHOFOESPROC) load(userptr, "glOrthofOES");
}
static void glad_gl_load_GL_OVR_multiview(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->OVR_multiview) return;
    context->FramebufferTextureMultiviewOVR = (PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVRPROC) load(userptr, "glFramebufferTextureMultiviewOVR");
}
static void glad_gl_load_GL_PGI_misc_hints(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->PGI_misc_hints) return;
    context->HintPGI = (PFNGLHINTPGIPROC) load(userptr, "glHintPGI");
}
static void glad_gl_load_GL_SGIS_detail_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_detail_texture) return;
    context->DetailTexFuncSGIS = (PFNGLDETAILTEXFUNCSGISPROC) load(userptr, "glDetailTexFuncSGIS");
    context->GetDetailTexFuncSGIS = (PFNGLGETDETAILTEXFUNCSGISPROC) load(userptr, "glGetDetailTexFuncSGIS");
}
static void glad_gl_load_GL_SGIS_fog_function(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_fog_function) return;
    context->FogFuncSGIS = (PFNGLFOGFUNCSGISPROC) load(userptr, "glFogFuncSGIS");
    context->GetFogFuncSGIS = (PFNGLGETFOGFUNCSGISPROC) load(userptr, "glGetFogFuncSGIS");
}
static void glad_gl_load_GL_SGIS_multisample(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_multisample) return;
    context->SampleMaskSGIS = (PFNGLSAMPLEMASKSGISPROC) load(userptr, "glSampleMaskSGIS");
    context->SamplePatternSGIS = (PFNGLSAMPLEPATTERNSGISPROC) load(userptr, "glSamplePatternSGIS");
}
static void glad_gl_load_GL_SGIS_pixel_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_pixel_texture) return;
    context->GetPixelTexGenParameterfvSGIS = (PFNGLGETPIXELTEXGENPARAMETERFVSGISPROC) load(userptr, "glGetPixelTexGenParameterfvSGIS");
    context->GetPixelTexGenParameterivSGIS = (PFNGLGETPIXELTEXGENPARAMETERIVSGISPROC) load(userptr, "glGetPixelTexGenParameterivSGIS");
    context->PixelTexGenParameterfSGIS = (PFNGLPIXELTEXGENPARAMETERFSGISPROC) load(userptr, "glPixelTexGenParameterfSGIS");
    context->PixelTexGenParameterfvSGIS = (PFNGLPIXELTEXGENPARAMETERFVSGISPROC) load(userptr, "glPixelTexGenParameterfvSGIS");
    context->PixelTexGenParameteriSGIS = (PFNGLPIXELTEXGENPARAMETERISGISPROC) load(userptr, "glPixelTexGenParameteriSGIS");
    context->PixelTexGenParameterivSGIS = (PFNGLPIXELTEXGENPARAMETERIVSGISPROC) load(userptr, "glPixelTexGenParameterivSGIS");
}
static void glad_gl_load_GL_SGIS_point_parameters(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_point_parameters) return;
    context->PointParameterfSGIS = (PFNGLPOINTPARAMETERFSGISPROC) load(userptr, "glPointParameterfSGIS");
    context->PointParameterfvSGIS = (PFNGLPOINTPARAMETERFVSGISPROC) load(userptr, "glPointParameterfvSGIS");
}
static void glad_gl_load_GL_SGIS_sharpen_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_sharpen_texture) return;
    context->GetSharpenTexFuncSGIS = (PFNGLGETSHARPENTEXFUNCSGISPROC) load(userptr, "glGetSharpenTexFuncSGIS");
    context->SharpenTexFuncSGIS = (PFNGLSHARPENTEXFUNCSGISPROC) load(userptr, "glSharpenTexFuncSGIS");
}
static void glad_gl_load_GL_SGIS_texture4D(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_texture4D) return;
    context->TexImage4DSGIS = (PFNGLTEXIMAGE4DSGISPROC) load(userptr, "glTexImage4DSGIS");
    context->TexSubImage4DSGIS = (PFNGLTEXSUBIMAGE4DSGISPROC) load(userptr, "glTexSubImage4DSGIS");
}
static void glad_gl_load_GL_SGIS_texture_color_mask(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_texture_color_mask) return;
    context->TextureColorMaskSGIS = (PFNGLTEXTURECOLORMASKSGISPROC) load(userptr, "glTextureColorMaskSGIS");
}
static void glad_gl_load_GL_SGIS_texture_filter4(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIS_texture_filter4) return;
    context->GetTexFilterFuncSGIS = (PFNGLGETTEXFILTERFUNCSGISPROC) load(userptr, "glGetTexFilterFuncSGIS");
    context->TexFilterFuncSGIS = (PFNGLTEXFILTERFUNCSGISPROC) load(userptr, "glTexFilterFuncSGIS");
}
static void glad_gl_load_GL_SGIX_async(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_async) return;
    context->AsyncMarkerSGIX = (PFNGLASYNCMARKERSGIXPROC) load(userptr, "glAsyncMarkerSGIX");
    context->DeleteAsyncMarkersSGIX = (PFNGLDELETEASYNCMARKERSSGIXPROC) load(userptr, "glDeleteAsyncMarkersSGIX");
    context->FinishAsyncSGIX = (PFNGLFINISHASYNCSGIXPROC) load(userptr, "glFinishAsyncSGIX");
    context->GenAsyncMarkersSGIX = (PFNGLGENASYNCMARKERSSGIXPROC) load(userptr, "glGenAsyncMarkersSGIX");
    context->IsAsyncMarkerSGIX = (PFNGLISASYNCMARKERSGIXPROC) load(userptr, "glIsAsyncMarkerSGIX");
    context->PollAsyncSGIX = (PFNGLPOLLASYNCSGIXPROC) load(userptr, "glPollAsyncSGIX");
}
static void glad_gl_load_GL_SGIX_flush_raster(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_flush_raster) return;
    context->FlushRasterSGIX = (PFNGLFLUSHRASTERSGIXPROC) load(userptr, "glFlushRasterSGIX");
}
static void glad_gl_load_GL_SGIX_fragment_lighting(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_fragment_lighting) return;
    context->FragmentColorMaterialSGIX = (PFNGLFRAGMENTCOLORMATERIALSGIXPROC) load(userptr, "glFragmentColorMaterialSGIX");
    context->FragmentLightModelfSGIX = (PFNGLFRAGMENTLIGHTMODELFSGIXPROC) load(userptr, "glFragmentLightModelfSGIX");
    context->FragmentLightModelfvSGIX = (PFNGLFRAGMENTLIGHTMODELFVSGIXPROC) load(userptr, "glFragmentLightModelfvSGIX");
    context->FragmentLightModeliSGIX = (PFNGLFRAGMENTLIGHTMODELISGIXPROC) load(userptr, "glFragmentLightModeliSGIX");
    context->FragmentLightModelivSGIX = (PFNGLFRAGMENTLIGHTMODELIVSGIXPROC) load(userptr, "glFragmentLightModelivSGIX");
    context->FragmentLightfSGIX = (PFNGLFRAGMENTLIGHTFSGIXPROC) load(userptr, "glFragmentLightfSGIX");
    context->FragmentLightfvSGIX = (PFNGLFRAGMENTLIGHTFVSGIXPROC) load(userptr, "glFragmentLightfvSGIX");
    context->FragmentLightiSGIX = (PFNGLFRAGMENTLIGHTISGIXPROC) load(userptr, "glFragmentLightiSGIX");
    context->FragmentLightivSGIX = (PFNGLFRAGMENTLIGHTIVSGIXPROC) load(userptr, "glFragmentLightivSGIX");
    context->FragmentMaterialfSGIX = (PFNGLFRAGMENTMATERIALFSGIXPROC) load(userptr, "glFragmentMaterialfSGIX");
    context->FragmentMaterialfvSGIX = (PFNGLFRAGMENTMATERIALFVSGIXPROC) load(userptr, "glFragmentMaterialfvSGIX");
    context->FragmentMaterialiSGIX = (PFNGLFRAGMENTMATERIALISGIXPROC) load(userptr, "glFragmentMaterialiSGIX");
    context->FragmentMaterialivSGIX = (PFNGLFRAGMENTMATERIALIVSGIXPROC) load(userptr, "glFragmentMaterialivSGIX");
    context->GetFragmentLightfvSGIX = (PFNGLGETFRAGMENTLIGHTFVSGIXPROC) load(userptr, "glGetFragmentLightfvSGIX");
    context->GetFragmentLightivSGIX = (PFNGLGETFRAGMENTLIGHTIVSGIXPROC) load(userptr, "glGetFragmentLightivSGIX");
    context->GetFragmentMaterialfvSGIX = (PFNGLGETFRAGMENTMATERIALFVSGIXPROC) load(userptr, "glGetFragmentMaterialfvSGIX");
    context->GetFragmentMaterialivSGIX = (PFNGLGETFRAGMENTMATERIALIVSGIXPROC) load(userptr, "glGetFragmentMaterialivSGIX");
    context->LightEnviSGIX = (PFNGLLIGHTENVISGIXPROC) load(userptr, "glLightEnviSGIX");
}
static void glad_gl_load_GL_SGIX_framezoom(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_framezoom) return;
    context->FrameZoomSGIX = (PFNGLFRAMEZOOMSGIXPROC) load(userptr, "glFrameZoomSGIX");
}
static void glad_gl_load_GL_SGIX_igloo_interface(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_igloo_interface) return;
    context->IglooInterfaceSGIX = (PFNGLIGLOOINTERFACESGIXPROC) load(userptr, "glIglooInterfaceSGIX");
}
static void glad_gl_load_GL_SGIX_instruments(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_instruments) return;
    context->GetInstrumentsSGIX = (PFNGLGETINSTRUMENTSSGIXPROC) load(userptr, "glGetInstrumentsSGIX");
    context->InstrumentsBufferSGIX = (PFNGLINSTRUMENTSBUFFERSGIXPROC) load(userptr, "glInstrumentsBufferSGIX");
    context->PollInstrumentsSGIX = (PFNGLPOLLINSTRUMENTSSGIXPROC) load(userptr, "glPollInstrumentsSGIX");
    context->ReadInstrumentsSGIX = (PFNGLREADINSTRUMENTSSGIXPROC) load(userptr, "glReadInstrumentsSGIX");
    context->StartInstrumentsSGIX = (PFNGLSTARTINSTRUMENTSSGIXPROC) load(userptr, "glStartInstrumentsSGIX");
    context->StopInstrumentsSGIX = (PFNGLSTOPINSTRUMENTSSGIXPROC) load(userptr, "glStopInstrumentsSGIX");
}
static void glad_gl_load_GL_SGIX_list_priority(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_list_priority) return;
    context->GetListParameterfvSGIX = (PFNGLGETLISTPARAMETERFVSGIXPROC) load(userptr, "glGetListParameterfvSGIX");
    context->GetListParameterivSGIX = (PFNGLGETLISTPARAMETERIVSGIXPROC) load(userptr, "glGetListParameterivSGIX");
    context->ListParameterfSGIX = (PFNGLLISTPARAMETERFSGIXPROC) load(userptr, "glListParameterfSGIX");
    context->ListParameterfvSGIX = (PFNGLLISTPARAMETERFVSGIXPROC) load(userptr, "glListParameterfvSGIX");
    context->ListParameteriSGIX = (PFNGLLISTPARAMETERISGIXPROC) load(userptr, "glListParameteriSGIX");
    context->ListParameterivSGIX = (PFNGLLISTPARAMETERIVSGIXPROC) load(userptr, "glListParameterivSGIX");
}
static void glad_gl_load_GL_SGIX_pixel_texture(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_pixel_texture) return;
    context->PixelTexGenSGIX = (PFNGLPIXELTEXGENSGIXPROC) load(userptr, "glPixelTexGenSGIX");
}
static void glad_gl_load_GL_SGIX_polynomial_ffd(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_polynomial_ffd) return;
    context->DeformSGIX = (PFNGLDEFORMSGIXPROC) load(userptr, "glDeformSGIX");
    context->DeformationMap3dSGIX = (PFNGLDEFORMATIONMAP3DSGIXPROC) load(userptr, "glDeformationMap3dSGIX");
    context->DeformationMap3fSGIX = (PFNGLDEFORMATIONMAP3FSGIXPROC) load(userptr, "glDeformationMap3fSGIX");
    context->LoadIdentityDeformationMapSGIX = (PFNGLLOADIDENTITYDEFORMATIONMAPSGIXPROC) load(userptr, "glLoadIdentityDeformationMapSGIX");
}
static void glad_gl_load_GL_SGIX_reference_plane(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_reference_plane) return;
    context->ReferencePlaneSGIX = (PFNGLREFERENCEPLANESGIXPROC) load(userptr, "glReferencePlaneSGIX");
}
static void glad_gl_load_GL_SGIX_sprite(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_sprite) return;
    context->SpriteParameterfSGIX = (PFNGLSPRITEPARAMETERFSGIXPROC) load(userptr, "glSpriteParameterfSGIX");
    context->SpriteParameterfvSGIX = (PFNGLSPRITEPARAMETERFVSGIXPROC) load(userptr, "glSpriteParameterfvSGIX");
    context->SpriteParameteriSGIX = (PFNGLSPRITEPARAMETERISGIXPROC) load(userptr, "glSpriteParameteriSGIX");
    context->SpriteParameterivSGIX = (PFNGLSPRITEPARAMETERIVSGIXPROC) load(userptr, "glSpriteParameterivSGIX");
}
static void glad_gl_load_GL_SGIX_tag_sample_buffer(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGIX_tag_sample_buffer) return;
    context->TagSampleBufferSGIX = (PFNGLTAGSAMPLEBUFFERSGIXPROC) load(userptr, "glTagSampleBufferSGIX");
}
static void glad_gl_load_GL_SGI_color_table(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SGI_color_table) return;
    context->ColorTableParameterfvSGI = (PFNGLCOLORTABLEPARAMETERFVSGIPROC) load(userptr, "glColorTableParameterfvSGI");
    context->ColorTableParameterivSGI = (PFNGLCOLORTABLEPARAMETERIVSGIPROC) load(userptr, "glColorTableParameterivSGI");
    context->ColorTableSGI = (PFNGLCOLORTABLESGIPROC) load(userptr, "glColorTableSGI");
    context->CopyColorTableSGI = (PFNGLCOPYCOLORTABLESGIPROC) load(userptr, "glCopyColorTableSGI");
    context->GetColorTableParameterfvSGI = (PFNGLGETCOLORTABLEPARAMETERFVSGIPROC) load(userptr, "glGetColorTableParameterfvSGI");
    context->GetColorTableParameterivSGI = (PFNGLGETCOLORTABLEPARAMETERIVSGIPROC) load(userptr, "glGetColorTableParameterivSGI");
    context->GetColorTableSGI = (PFNGLGETCOLORTABLESGIPROC) load(userptr, "glGetColorTableSGI");
}
static void glad_gl_load_GL_SUNX_constant_data(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SUNX_constant_data) return;
    context->FinishTextureSUNX = (PFNGLFINISHTEXTURESUNXPROC) load(userptr, "glFinishTextureSUNX");
}
static void glad_gl_load_GL_SUN_global_alpha(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SUN_global_alpha) return;
    context->GlobalAlphaFactorbSUN = (PFNGLGLOBALALPHAFACTORBSUNPROC) load(userptr, "glGlobalAlphaFactorbSUN");
    context->GlobalAlphaFactordSUN = (PFNGLGLOBALALPHAFACTORDSUNPROC) load(userptr, "glGlobalAlphaFactordSUN");
    context->GlobalAlphaFactorfSUN = (PFNGLGLOBALALPHAFACTORFSUNPROC) load(userptr, "glGlobalAlphaFactorfSUN");
    context->GlobalAlphaFactoriSUN = (PFNGLGLOBALALPHAFACTORISUNPROC) load(userptr, "glGlobalAlphaFactoriSUN");
    context->GlobalAlphaFactorsSUN = (PFNGLGLOBALALPHAFACTORSSUNPROC) load(userptr, "glGlobalAlphaFactorsSUN");
    context->GlobalAlphaFactorubSUN = (PFNGLGLOBALALPHAFACTORUBSUNPROC) load(userptr, "glGlobalAlphaFactorubSUN");
    context->GlobalAlphaFactoruiSUN = (PFNGLGLOBALALPHAFACTORUISUNPROC) load(userptr, "glGlobalAlphaFactoruiSUN");
    context->GlobalAlphaFactorusSUN = (PFNGLGLOBALALPHAFACTORUSSUNPROC) load(userptr, "glGlobalAlphaFactorusSUN");
}
static void glad_gl_load_GL_SUN_mesh_array(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SUN_mesh_array) return;
    context->DrawMeshArraysSUN = (PFNGLDRAWMESHARRAYSSUNPROC) load(userptr, "glDrawMeshArraysSUN");
}
static void glad_gl_load_GL_SUN_triangle_list(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SUN_triangle_list) return;
    context->ReplacementCodePointerSUN = (PFNGLREPLACEMENTCODEPOINTERSUNPROC) load(userptr, "glReplacementCodePointerSUN");
    context->ReplacementCodeubSUN = (PFNGLREPLACEMENTCODEUBSUNPROC) load(userptr, "glReplacementCodeubSUN");
    context->ReplacementCodeubvSUN = (PFNGLREPLACEMENTCODEUBVSUNPROC) load(userptr, "glReplacementCodeubvSUN");
    context->ReplacementCodeuiSUN = (PFNGLREPLACEMENTCODEUISUNPROC) load(userptr, "glReplacementCodeuiSUN");
    context->ReplacementCodeuivSUN = (PFNGLREPLACEMENTCODEUIVSUNPROC) load(userptr, "glReplacementCodeuivSUN");
    context->ReplacementCodeusSUN = (PFNGLREPLACEMENTCODEUSSUNPROC) load(userptr, "glReplacementCodeusSUN");
    context->ReplacementCodeusvSUN = (PFNGLREPLACEMENTCODEUSVSUNPROC) load(userptr, "glReplacementCodeusvSUN");
}
static void glad_gl_load_GL_SUN_vertex(GladGLContext *context, GLADuserptrloadfunc load, void* userptr) {
    if(!context->SUN_vertex) return;
    context->Color3fVertex3fSUN = (PFNGLCOLOR3FVERTEX3FSUNPROC) load(userptr, "glColor3fVertex3fSUN");
    context->Color3fVertex3fvSUN = (PFNGLCOLOR3FVERTEX3FVSUNPROC) load(userptr, "glColor3fVertex3fvSUN");
    context->Color4fNormal3fVertex3fSUN = (PFNGLCOLOR4FNORMAL3FVERTEX3FSUNPROC) load(userptr, "glColor4fNormal3fVertex3fSUN");
    context->Color4fNormal3fVertex3fvSUN = (PFNGLCOLOR4FNORMAL3FVERTEX3FVSUNPROC) load(userptr, "glColor4fNormal3fVertex3fvSUN");
    context->Color4ubVertex2fSUN = (PFNGLCOLOR4UBVERTEX2FSUNPROC) load(userptr, "glColor4ubVertex2fSUN");
    context->Color4ubVertex2fvSUN = (PFNGLCOLOR4UBVERTEX2FVSUNPROC) load(userptr, "glColor4ubVertex2fvSUN");
    context->Color4ubVertex3fSUN = (PFNGLCOLOR4UBVERTEX3FSUNPROC) load(userptr, "glColor4ubVertex3fSUN");
    context->Color4ubVertex3fvSUN = (PFNGLCOLOR4UBVERTEX3FVSUNPROC) load(userptr, "glColor4ubVertex3fvSUN");
    context->Normal3fVertex3fSUN = (PFNGLNORMAL3FVERTEX3FSUNPROC) load(userptr, "glNormal3fVertex3fSUN");
    context->Normal3fVertex3fvSUN = (PFNGLNORMAL3FVERTEX3FVSUNPROC) load(userptr, "glNormal3fVertex3fvSUN");
    context->ReplacementCodeuiColor3fVertex3fSUN = (PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiColor3fVertex3fSUN");
    context->ReplacementCodeuiColor3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiColor3fVertex3fvSUN");
    context->ReplacementCodeuiColor4fNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiColor4fNormal3fVertex3fSUN");
    context->ReplacementCodeuiColor4fNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiColor4fNormal3fVertex3fvSUN");
    context->ReplacementCodeuiColor4ubVertex3fSUN = (PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiColor4ubVertex3fSUN");
    context->ReplacementCodeuiColor4ubVertex3fvSUN = (PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiColor4ubVertex3fvSUN");
    context->ReplacementCodeuiNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiNormal3fVertex3fSUN");
    context->ReplacementCodeuiNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiNormal3fVertex3fvSUN");
    context->ReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN");
    context->ReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN");
    context->ReplacementCodeuiTexCoord2fNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiTexCoord2fNormal3fVertex3fSUN");
    context->ReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN");
    context->ReplacementCodeuiTexCoord2fVertex3fSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiTexCoord2fVertex3fSUN");
    context->ReplacementCodeuiTexCoord2fVertex3fvSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiTexCoord2fVertex3fvSUN");
    context->ReplacementCodeuiVertex3fSUN = (PFNGLREPLACEMENTCODEUIVERTEX3FSUNPROC) load(userptr, "glReplacementCodeuiVertex3fSUN");
    context->ReplacementCodeuiVertex3fvSUN = (PFNGLREPLACEMENTCODEUIVERTEX3FVSUNPROC) load(userptr, "glReplacementCodeuiVertex3fvSUN");
    context->TexCoord2fColor3fVertex3fSUN = (PFNGLTEXCOORD2FCOLOR3FVERTEX3FSUNPROC) load(userptr, "glTexCoord2fColor3fVertex3fSUN");
    context->TexCoord2fColor3fVertex3fvSUN = (PFNGLTEXCOORD2FCOLOR3FVERTEX3FVSUNPROC) load(userptr, "glTexCoord2fColor3fVertex3fvSUN");
    context->TexCoord2fColor4fNormal3fVertex3fSUN = (PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC) load(userptr, "glTexCoord2fColor4fNormal3fVertex3fSUN");
    context->TexCoord2fColor4fNormal3fVertex3fvSUN = (PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC) load(userptr, "glTexCoord2fColor4fNormal3fVertex3fvSUN");
    context->TexCoord2fColor4ubVertex3fSUN = (PFNGLTEXCOORD2FCOLOR4UBVERTEX3FSUNPROC) load(userptr, "glTexCoord2fColor4ubVertex3fSUN");
    context->TexCoord2fColor4ubVertex3fvSUN = (PFNGLTEXCOORD2FCOLOR4UBVERTEX3FVSUNPROC) load(userptr, "glTexCoord2fColor4ubVertex3fvSUN");
    context->TexCoord2fNormal3fVertex3fSUN = (PFNGLTEXCOORD2FNORMAL3FVERTEX3FSUNPROC) load(userptr, "glTexCoord2fNormal3fVertex3fSUN");
    context->TexCoord2fNormal3fVertex3fvSUN = (PFNGLTEXCOORD2FNORMAL3FVERTEX3FVSUNPROC) load(userptr, "glTexCoord2fNormal3fVertex3fvSUN");
    context->TexCoord2fVertex3fSUN = (PFNGLTEXCOORD2FVERTEX3FSUNPROC) load(userptr, "glTexCoord2fVertex3fSUN");
    context->TexCoord2fVertex3fvSUN = (PFNGLTEXCOORD2FVERTEX3FVSUNPROC) load(userptr, "glTexCoord2fVertex3fvSUN");
    context->TexCoord4fColor4fNormal3fVertex4fSUN = (PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FSUNPROC) load(userptr, "glTexCoord4fColor4fNormal3fVertex4fSUN");
    context->TexCoord4fColor4fNormal3fVertex4fvSUN = (PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FVSUNPROC) load(userptr, "glTexCoord4fColor4fNormal3fVertex4fvSUN");
    context->TexCoord4fVertex4fSUN = (PFNGLTEXCOORD4FVERTEX4FSUNPROC) load(userptr, "glTexCoord4fVertex4fSUN");
    context->TexCoord4fVertex4fvSUN = (PFNGLTEXCOORD4FVERTEX4FVSUNPROC) load(userptr, "glTexCoord4fVertex4fvSUN");
}



#if defined(GL_ES_VERSION_3_0) || defined(GL_VERSION_3_0)
#define GLAD_GL_IS_SOME_NEW_VERSION 1
#else
#define GLAD_GL_IS_SOME_NEW_VERSION 0
#endif

static int glad_gl_get_extensions(GladGLContext *context, int version, const char **out_exts, unsigned int *out_num_exts_i, char ***out_exts_i) {
#if GLAD_GL_IS_SOME_NEW_VERSION
    if(GLAD_VERSION_MAJOR(version) < 3) {
#else
    GLAD_UNUSED(version);
    GLAD_UNUSED(out_num_exts_i);
    GLAD_UNUSED(out_exts_i);
#endif
        if (context->GetString == NULL) {
            return 0;
        }
        *out_exts = (const char *)context->GetString(GL_EXTENSIONS);
#if GLAD_GL_IS_SOME_NEW_VERSION
    } else {
        unsigned int index = 0;
        unsigned int num_exts_i = 0;
        char **exts_i = NULL;
        if (context->GetStringi == NULL || context->GetIntegerv == NULL) {
            return 0;
        }
        context->GetIntegerv(GL_NUM_EXTENSIONS, (int*) &num_exts_i);
        if (num_exts_i > 0) {
            exts_i = (char **) malloc(num_exts_i * (sizeof *exts_i));
        }
        if (exts_i == NULL) {
            return 0;
        }
        for(index = 0; index < num_exts_i; index++) {
            const char *gl_str_tmp = (const char*) context->GetStringi(GL_EXTENSIONS, index);
            size_t len = strlen(gl_str_tmp) + 1;

            char *local_str = (char*) malloc(len * sizeof(char));
            if(local_str != NULL) {
                memcpy(local_str, gl_str_tmp, len * sizeof(char));
            }

            exts_i[index] = local_str;
        }

        *out_num_exts_i = num_exts_i;
        *out_exts_i = exts_i;
    }
#endif
    return 1;
}
static void glad_gl_free_extensions(char **exts_i, unsigned int num_exts_i) {
    if (exts_i != NULL) {
        unsigned int index;
        for(index = 0; index < num_exts_i; index++) {
            free((void *) (exts_i[index]));
        }
        free((void *)exts_i);
        exts_i = NULL;
    }
}
static int glad_gl_has_extension(int version, const char *exts, unsigned int num_exts_i, char **exts_i, const char *ext) {
    if(GLAD_VERSION_MAJOR(version) < 3 || !GLAD_GL_IS_SOME_NEW_VERSION) {
        const char *extensions;
        const char *loc;
        const char *terminator;
        extensions = exts;
        if(extensions == NULL || ext == NULL) {
            return 0;
        }
        while(1) {
            loc = strstr(extensions, ext);
            if(loc == NULL) {
                return 0;
            }
            terminator = loc + strlen(ext);
            if((loc == extensions || *(loc - 1) == ' ') &&
                (*terminator == ' ' || *terminator == '\0')) {
                return 1;
            }
            extensions = terminator;
        }
    } else {
        unsigned int index;
        for(index = 0; index < num_exts_i; index++) {
            const char *e = exts_i[index];
            if(strcmp(e, ext) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

static GLADapiproc glad_gl_get_proc_from_userptr(void *userptr, const char* name) {
    return (GLAD_GNUC_EXTENSION (GLADapiproc (*)(const char *name)) userptr)(name);
}

static int glad_gl_find_extensions_gl(GladGLContext *context, int version) {
    const char *exts = NULL;
    unsigned int num_exts_i = 0;
    char **exts_i = NULL;
    if (!glad_gl_get_extensions(context, version, &exts, &num_exts_i, &exts_i)) return 0;

    context->_3DFX_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_3DFX_multisample");
    context->_3DFX_tbuffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_3DFX_tbuffer");
    context->_3DFX_texture_compression_FXT1 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_3DFX_texture_compression_FXT1");
    context->AMD_blend_minmax_factor = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_blend_minmax_factor");
    context->AMD_conservative_depth = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_conservative_depth");
    context->AMD_debug_output = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_debug_output");
    context->AMD_depth_clamp_separate = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_depth_clamp_separate");
    context->AMD_draw_buffers_blend = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_draw_buffers_blend");
    context->AMD_framebuffer_multisample_advanced = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_framebuffer_multisample_advanced");
    context->AMD_framebuffer_sample_positions = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_framebuffer_sample_positions");
    context->AMD_gcn_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_gcn_shader");
    context->AMD_gpu_shader_half_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_gpu_shader_half_float");
    context->AMD_gpu_shader_int16 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_gpu_shader_int16");
    context->AMD_gpu_shader_int64 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_gpu_shader_int64");
    context->AMD_interleaved_elements = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_interleaved_elements");
    context->AMD_multi_draw_indirect = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_multi_draw_indirect");
    context->AMD_name_gen_delete = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_name_gen_delete");
    context->AMD_occlusion_query_event = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_occlusion_query_event");
    context->AMD_performance_monitor = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_performance_monitor");
    context->AMD_pinned_memory = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_pinned_memory");
    context->AMD_query_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_query_buffer_object");
    context->AMD_sample_positions = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_sample_positions");
    context->AMD_seamless_cubemap_per_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_seamless_cubemap_per_texture");
    context->AMD_shader_atomic_counter_ops = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_shader_atomic_counter_ops");
    context->AMD_shader_ballot = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_shader_ballot");
    context->AMD_shader_explicit_vertex_parameter = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_shader_explicit_vertex_parameter");
    context->AMD_shader_gpu_shader_half_float_fetch = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_shader_gpu_shader_half_float_fetch");
    context->AMD_shader_image_load_store_lod = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_shader_image_load_store_lod");
    context->AMD_shader_stencil_export = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_shader_stencil_export");
    context->AMD_shader_trinary_minmax = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_shader_trinary_minmax");
    context->AMD_sparse_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_sparse_texture");
    context->AMD_stencil_operation_extended = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_stencil_operation_extended");
    context->AMD_texture_gather_bias_lod = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_texture_gather_bias_lod");
    context->AMD_texture_texture4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_texture_texture4");
    context->AMD_transform_feedback3_lines_triangles = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_transform_feedback3_lines_triangles");
    context->AMD_transform_feedback4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_transform_feedback4");
    context->AMD_vertex_shader_layer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_vertex_shader_layer");
    context->AMD_vertex_shader_tessellator = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_vertex_shader_tessellator");
    context->AMD_vertex_shader_viewport_index = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_AMD_vertex_shader_viewport_index");
    context->APPLE_aux_depth_stencil = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_aux_depth_stencil");
    context->APPLE_client_storage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_client_storage");
    context->APPLE_element_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_element_array");
    context->APPLE_fence = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_fence");
    context->APPLE_float_pixels = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_float_pixels");
    context->APPLE_flush_buffer_range = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_flush_buffer_range");
    context->APPLE_object_purgeable = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_object_purgeable");
    context->APPLE_rgb_422 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_rgb_422");
    context->APPLE_row_bytes = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_row_bytes");
    context->APPLE_specular_vector = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_specular_vector");
    context->APPLE_texture_range = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_texture_range");
    context->APPLE_transform_hint = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_transform_hint");
    context->APPLE_vertex_array_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_vertex_array_object");
    context->APPLE_vertex_array_range = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_vertex_array_range");
    context->APPLE_vertex_program_evaluators = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_vertex_program_evaluators");
    context->APPLE_ycbcr_422 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_APPLE_ycbcr_422");
    context->ARB_ES2_compatibility = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_ES2_compatibility");
    context->ARB_ES3_1_compatibility = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_ES3_1_compatibility");
    context->ARB_ES3_2_compatibility = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_ES3_2_compatibility");
    context->ARB_ES3_compatibility = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_ES3_compatibility");
    context->ARB_arrays_of_arrays = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_arrays_of_arrays");
    context->ARB_base_instance = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_base_instance");
    context->ARB_bindless_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_bindless_texture");
    context->ARB_blend_func_extended = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_blend_func_extended");
    context->ARB_buffer_storage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_buffer_storage");
    context->ARB_cl_event = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_cl_event");
    context->ARB_clear_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_clear_buffer_object");
    context->ARB_clear_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_clear_texture");
    context->ARB_clip_control = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_clip_control");
    context->ARB_color_buffer_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_color_buffer_float");
    context->ARB_compatibility = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_compatibility");
    context->ARB_compressed_texture_pixel_storage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_compressed_texture_pixel_storage");
    context->ARB_compute_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_compute_shader");
    context->ARB_compute_variable_group_size = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_compute_variable_group_size");
    context->ARB_conditional_render_inverted = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_conditional_render_inverted");
    context->ARB_conservative_depth = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_conservative_depth");
    context->ARB_copy_buffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_copy_buffer");
    context->ARB_copy_image = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_copy_image");
    context->ARB_cull_distance = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_cull_distance");
    context->ARB_debug_output = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_debug_output");
    context->ARB_depth_buffer_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_depth_buffer_float");
    context->ARB_depth_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_depth_clamp");
    context->ARB_depth_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_depth_texture");
    context->ARB_derivative_control = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_derivative_control");
    context->ARB_direct_state_access = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_direct_state_access");
    context->ARB_draw_buffers = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_draw_buffers");
    context->ARB_draw_buffers_blend = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_draw_buffers_blend");
    context->ARB_draw_elements_base_vertex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_draw_elements_base_vertex");
    context->ARB_draw_indirect = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_draw_indirect");
    context->ARB_draw_instanced = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_draw_instanced");
    context->ARB_enhanced_layouts = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_enhanced_layouts");
    context->ARB_explicit_attrib_location = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_explicit_attrib_location");
    context->ARB_explicit_uniform_location = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_explicit_uniform_location");
    context->ARB_fragment_coord_conventions = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_fragment_coord_conventions");
    context->ARB_fragment_layer_viewport = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_fragment_layer_viewport");
    context->ARB_fragment_program = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_fragment_program");
    context->ARB_fragment_program_shadow = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_fragment_program_shadow");
    context->ARB_fragment_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_fragment_shader");
    context->ARB_fragment_shader_interlock = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_fragment_shader_interlock");
    context->ARB_framebuffer_no_attachments = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_framebuffer_no_attachments");
    context->ARB_framebuffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_framebuffer_object");
    context->ARB_framebuffer_sRGB = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_framebuffer_sRGB");
    context->ARB_geometry_shader4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_geometry_shader4");
    context->ARB_get_program_binary = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_get_program_binary");
    context->ARB_get_texture_sub_image = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_get_texture_sub_image");
    context->ARB_gl_spirv = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_gl_spirv");
    context->ARB_gpu_shader5 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_gpu_shader5");
    context->ARB_gpu_shader_fp64 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_gpu_shader_fp64");
    context->ARB_gpu_shader_int64 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_gpu_shader_int64");
    context->ARB_half_float_pixel = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_half_float_pixel");
    context->ARB_half_float_vertex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_half_float_vertex");
    context->ARB_imaging = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_imaging");
    context->ARB_indirect_parameters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_indirect_parameters");
    context->ARB_instanced_arrays = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_instanced_arrays");
    context->ARB_internalformat_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_internalformat_query");
    context->ARB_internalformat_query2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_internalformat_query2");
    context->ARB_invalidate_subdata = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_invalidate_subdata");
    context->ARB_map_buffer_alignment = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_map_buffer_alignment");
    context->ARB_map_buffer_range = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_map_buffer_range");
    context->ARB_matrix_palette = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_matrix_palette");
    context->ARB_multi_bind = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_multi_bind");
    context->ARB_multi_draw_indirect = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_multi_draw_indirect");
    context->ARB_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_multisample");
    context->ARB_multitexture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_multitexture");
    context->ARB_occlusion_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_occlusion_query");
    context->ARB_occlusion_query2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_occlusion_query2");
    context->ARB_parallel_shader_compile = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_parallel_shader_compile");
    context->ARB_pipeline_statistics_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_pipeline_statistics_query");
    context->ARB_pixel_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_pixel_buffer_object");
    context->ARB_point_parameters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_point_parameters");
    context->ARB_point_sprite = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_point_sprite");
    context->ARB_polygon_offset_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_polygon_offset_clamp");
    context->ARB_post_depth_coverage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_post_depth_coverage");
    context->ARB_program_interface_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_program_interface_query");
    context->ARB_provoking_vertex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_provoking_vertex");
    context->ARB_query_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_query_buffer_object");
    context->ARB_robust_buffer_access_behavior = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_robust_buffer_access_behavior");
    context->ARB_robustness = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_robustness");
    context->ARB_robustness_isolation = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_robustness_isolation");
    context->ARB_sample_locations = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sample_locations");
    context->ARB_sample_shading = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sample_shading");
    context->ARB_sampler_objects = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sampler_objects");
    context->ARB_seamless_cube_map = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_seamless_cube_map");
    context->ARB_seamless_cubemap_per_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_seamless_cubemap_per_texture");
    context->ARB_separate_shader_objects = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_separate_shader_objects");
    context->ARB_shader_atomic_counter_ops = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_atomic_counter_ops");
    context->ARB_shader_atomic_counters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_atomic_counters");
    context->ARB_shader_ballot = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_ballot");
    context->ARB_shader_bit_encoding = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_bit_encoding");
    context->ARB_shader_clock = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_clock");
    context->ARB_shader_draw_parameters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_draw_parameters");
    context->ARB_shader_group_vote = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_group_vote");
    context->ARB_shader_image_load_store = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_image_load_store");
    context->ARB_shader_image_size = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_image_size");
    context->ARB_shader_objects = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_objects");
    context->ARB_shader_precision = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_precision");
    context->ARB_shader_stencil_export = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_stencil_export");
    context->ARB_shader_storage_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_storage_buffer_object");
    context->ARB_shader_subroutine = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_subroutine");
    context->ARB_shader_texture_image_samples = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_texture_image_samples");
    context->ARB_shader_texture_lod = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_texture_lod");
    context->ARB_shader_viewport_layer_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shader_viewport_layer_array");
    context->ARB_shading_language_100 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shading_language_100");
    context->ARB_shading_language_420pack = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shading_language_420pack");
    context->ARB_shading_language_include = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shading_language_include");
    context->ARB_shading_language_packing = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shading_language_packing");
    context->ARB_shadow = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shadow");
    context->ARB_shadow_ambient = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_shadow_ambient");
    context->ARB_sparse_buffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sparse_buffer");
    context->ARB_sparse_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sparse_texture");
    context->ARB_sparse_texture2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sparse_texture2");
    context->ARB_sparse_texture_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sparse_texture_clamp");
    context->ARB_spirv_extensions = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_spirv_extensions");
    context->ARB_stencil_texturing = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_stencil_texturing");
    context->ARB_sync = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_sync");
    context->ARB_tessellation_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_tessellation_shader");
    context->ARB_texture_barrier = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_barrier");
    context->ARB_texture_border_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_border_clamp");
    context->ARB_texture_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_buffer_object");
    context->ARB_texture_buffer_object_rgb32 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_buffer_object_rgb32");
    context->ARB_texture_buffer_range = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_buffer_range");
    context->ARB_texture_compression = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_compression");
    context->ARB_texture_compression_bptc = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_compression_bptc");
    context->ARB_texture_compression_rgtc = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_compression_rgtc");
    context->ARB_texture_cube_map = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_cube_map");
    context->ARB_texture_cube_map_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_cube_map_array");
    context->ARB_texture_env_add = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_env_add");
    context->ARB_texture_env_combine = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_env_combine");
    context->ARB_texture_env_crossbar = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_env_crossbar");
    context->ARB_texture_env_dot3 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_env_dot3");
    context->ARB_texture_filter_anisotropic = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_filter_anisotropic");
    context->ARB_texture_filter_minmax = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_filter_minmax");
    context->ARB_texture_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_float");
    context->ARB_texture_gather = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_gather");
    context->ARB_texture_mirror_clamp_to_edge = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_mirror_clamp_to_edge");
    context->ARB_texture_mirrored_repeat = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_mirrored_repeat");
    context->ARB_texture_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_multisample");
    context->ARB_texture_non_power_of_two = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_non_power_of_two");
    context->ARB_texture_query_levels = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_query_levels");
    context->ARB_texture_query_lod = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_query_lod");
    context->ARB_texture_rectangle = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_rectangle");
    context->ARB_texture_rg = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_rg");
    context->ARB_texture_rgb10_a2ui = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_rgb10_a2ui");
    context->ARB_texture_stencil8 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_stencil8");
    context->ARB_texture_storage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_storage");
    context->ARB_texture_storage_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_storage_multisample");
    context->ARB_texture_swizzle = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_swizzle");
    context->ARB_texture_view = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_texture_view");
    context->ARB_timer_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_timer_query");
    context->ARB_transform_feedback2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_transform_feedback2");
    context->ARB_transform_feedback3 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_transform_feedback3");
    context->ARB_transform_feedback_instanced = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_transform_feedback_instanced");
    context->ARB_transform_feedback_overflow_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_transform_feedback_overflow_query");
    context->ARB_transpose_matrix = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_transpose_matrix");
    context->ARB_uniform_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_uniform_buffer_object");
    context->ARB_vertex_array_bgra = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_array_bgra");
    context->ARB_vertex_array_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_array_object");
    context->ARB_vertex_attrib_64bit = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_attrib_64bit");
    context->ARB_vertex_attrib_binding = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_attrib_binding");
    context->ARB_vertex_blend = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_blend");
    context->ARB_vertex_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_buffer_object");
    context->ARB_vertex_program = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_program");
    context->ARB_vertex_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_shader");
    context->ARB_vertex_type_10f_11f_11f_rev = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_type_10f_11f_11f_rev");
    context->ARB_vertex_type_2_10_10_10_rev = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_vertex_type_2_10_10_10_rev");
    context->ARB_viewport_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_viewport_array");
    context->ARB_window_pos = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ARB_window_pos");
    context->ATI_draw_buffers = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_draw_buffers");
    context->ATI_element_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_element_array");
    context->ATI_envmap_bumpmap = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_envmap_bumpmap");
    context->ATI_fragment_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_fragment_shader");
    context->ATI_map_object_buffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_map_object_buffer");
    context->ATI_meminfo = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_meminfo");
    context->ATI_pixel_format_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_pixel_format_float");
    context->ATI_pn_triangles = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_pn_triangles");
    context->ATI_separate_stencil = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_separate_stencil");
    context->ATI_text_fragment_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_text_fragment_shader");
    context->ATI_texture_env_combine3 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_texture_env_combine3");
    context->ATI_texture_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_texture_float");
    context->ATI_texture_mirror_once = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_texture_mirror_once");
    context->ATI_vertex_array_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_vertex_array_object");
    context->ATI_vertex_attrib_array_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_vertex_attrib_array_object");
    context->ATI_vertex_streams = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_ATI_vertex_streams");
    context->EXT_422_pixels = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_422_pixels");
    context->EXT_EGL_image_storage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_EGL_image_storage");
    context->EXT_EGL_sync = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_EGL_sync");
    context->EXT_abgr = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_abgr");
    context->EXT_bgra = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_bgra");
    context->EXT_bindable_uniform = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_bindable_uniform");
    context->EXT_blend_color = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_blend_color");
    context->EXT_blend_equation_separate = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_blend_equation_separate");
    context->EXT_blend_func_separate = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_blend_func_separate");
    context->EXT_blend_logic_op = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_blend_logic_op");
    context->EXT_blend_minmax = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_blend_minmax");
    context->EXT_blend_subtract = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_blend_subtract");
    context->EXT_clip_volume_hint = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_clip_volume_hint");
    context->EXT_cmyka = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_cmyka");
    context->EXT_color_subtable = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_color_subtable");
    context->EXT_compiled_vertex_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_compiled_vertex_array");
    context->EXT_convolution = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_convolution");
    context->EXT_coordinate_frame = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_coordinate_frame");
    context->EXT_copy_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_copy_texture");
    context->EXT_cull_vertex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_cull_vertex");
    context->EXT_debug_label = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_debug_label");
    context->EXT_debug_marker = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_debug_marker");
    context->EXT_depth_bounds_test = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_depth_bounds_test");
    context->EXT_direct_state_access = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_direct_state_access");
    context->EXT_draw_buffers2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_draw_buffers2");
    context->EXT_draw_instanced = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_draw_instanced");
    context->EXT_draw_range_elements = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_draw_range_elements");
    context->EXT_external_buffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_external_buffer");
    context->EXT_fog_coord = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_fog_coord");
    context->EXT_framebuffer_blit = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_framebuffer_blit");
    context->EXT_framebuffer_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_framebuffer_multisample");
    context->EXT_framebuffer_multisample_blit_scaled = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_framebuffer_multisample_blit_scaled");
    context->EXT_framebuffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_framebuffer_object");
    context->EXT_framebuffer_sRGB = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_framebuffer_sRGB");
    context->EXT_geometry_shader4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_geometry_shader4");
    context->EXT_gpu_program_parameters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_gpu_program_parameters");
    context->EXT_gpu_shader4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_gpu_shader4");
    context->EXT_histogram = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_histogram");
    context->EXT_index_array_formats = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_index_array_formats");
    context->EXT_index_func = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_index_func");
    context->EXT_index_material = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_index_material");
    context->EXT_index_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_index_texture");
    context->EXT_light_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_light_texture");
    context->EXT_memory_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_memory_object");
    context->EXT_memory_object_fd = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_memory_object_fd");
    context->EXT_memory_object_win32 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_memory_object_win32");
    context->EXT_misc_attribute = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_misc_attribute");
    context->EXT_multi_draw_arrays = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_multi_draw_arrays");
    context->EXT_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_multisample");
    context->EXT_multiview_tessellation_geometry_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_multiview_tessellation_geometry_shader");
    context->EXT_multiview_texture_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_multiview_texture_multisample");
    context->EXT_multiview_timer_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_multiview_timer_query");
    context->EXT_packed_depth_stencil = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_packed_depth_stencil");
    context->EXT_packed_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_packed_float");
    context->EXT_packed_pixels = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_packed_pixels");
    context->EXT_paletted_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_paletted_texture");
    context->EXT_pixel_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_pixel_buffer_object");
    context->EXT_pixel_transform = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_pixel_transform");
    context->EXT_pixel_transform_color_table = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_pixel_transform_color_table");
    context->EXT_point_parameters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_point_parameters");
    context->EXT_polygon_offset = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_polygon_offset");
    context->EXT_polygon_offset_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_polygon_offset_clamp");
    context->EXT_post_depth_coverage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_post_depth_coverage");
    context->EXT_provoking_vertex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_provoking_vertex");
    context->EXT_raster_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_raster_multisample");
    context->EXT_rescale_normal = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_rescale_normal");
    context->EXT_secondary_color = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_secondary_color");
    context->EXT_semaphore = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_semaphore");
    context->EXT_semaphore_fd = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_semaphore_fd");
    context->EXT_semaphore_win32 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_semaphore_win32");
    context->EXT_separate_shader_objects = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_separate_shader_objects");
    context->EXT_separate_specular_color = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_separate_specular_color");
    context->EXT_shader_framebuffer_fetch = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shader_framebuffer_fetch");
    context->EXT_shader_framebuffer_fetch_non_coherent = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shader_framebuffer_fetch_non_coherent");
    context->EXT_shader_image_load_formatted = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shader_image_load_formatted");
    context->EXT_shader_image_load_store = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shader_image_load_store");
    context->EXT_shader_integer_mix = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shader_integer_mix");
    context->EXT_shader_samples_identical = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shader_samples_identical");
    context->EXT_shadow_funcs = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shadow_funcs");
    context->EXT_shared_texture_palette = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_shared_texture_palette");
    context->EXT_sparse_texture2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_sparse_texture2");
    context->EXT_stencil_clear_tag = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_stencil_clear_tag");
    context->EXT_stencil_two_side = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_stencil_two_side");
    context->EXT_stencil_wrap = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_stencil_wrap");
    context->EXT_subtexture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_subtexture");
    context->EXT_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture");
    context->EXT_texture3D = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture3D");
    context->EXT_texture_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_array");
    context->EXT_texture_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_buffer_object");
    context->EXT_texture_compression_latc = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_compression_latc");
    context->EXT_texture_compression_rgtc = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_compression_rgtc");
    context->EXT_texture_compression_s3tc = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_compression_s3tc");
    context->EXT_texture_cube_map = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_cube_map");
    context->EXT_texture_env_add = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_env_add");
    context->EXT_texture_env_combine = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_env_combine");
    context->EXT_texture_env_dot3 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_env_dot3");
    context->EXT_texture_filter_anisotropic = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_filter_anisotropic");
    context->EXT_texture_filter_minmax = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_filter_minmax");
    context->EXT_texture_integer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_integer");
    context->EXT_texture_lod_bias = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_lod_bias");
    context->EXT_texture_mirror_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_mirror_clamp");
    context->EXT_texture_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_object");
    context->EXT_texture_perturb_normal = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_perturb_normal");
    context->EXT_texture_sRGB = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_sRGB");
    context->EXT_texture_sRGB_R8 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_sRGB_R8");
    context->EXT_texture_sRGB_RG8 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_sRGB_RG8");
    context->EXT_texture_sRGB_decode = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_sRGB_decode");
    context->EXT_texture_shadow_lod = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_shadow_lod");
    context->EXT_texture_shared_exponent = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_shared_exponent");
    context->EXT_texture_snorm = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_snorm");
    context->EXT_texture_storage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_storage");
    context->EXT_texture_swizzle = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_texture_swizzle");
    context->EXT_timer_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_timer_query");
    context->EXT_transform_feedback = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_transform_feedback");
    context->EXT_vertex_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_vertex_array");
    context->EXT_vertex_array_bgra = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_vertex_array_bgra");
    context->EXT_vertex_attrib_64bit = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_vertex_attrib_64bit");
    context->EXT_vertex_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_vertex_shader");
    context->EXT_vertex_weighting = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_vertex_weighting");
    context->EXT_win32_keyed_mutex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_win32_keyed_mutex");
    context->EXT_window_rectangles = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_window_rectangles");
    context->EXT_x11_sync_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_EXT_x11_sync_object");
    context->GREMEDY_frame_terminator = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_GREMEDY_frame_terminator");
    context->GREMEDY_string_marker = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_GREMEDY_string_marker");
    context->HP_convolution_border_modes = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_HP_convolution_border_modes");
    context->HP_image_transform = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_HP_image_transform");
    context->HP_occlusion_test = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_HP_occlusion_test");
    context->HP_texture_lighting = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_HP_texture_lighting");
    context->IBM_cull_vertex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_IBM_cull_vertex");
    context->IBM_multimode_draw_arrays = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_IBM_multimode_draw_arrays");
    context->IBM_rasterpos_clip = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_IBM_rasterpos_clip");
    context->IBM_static_data = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_IBM_static_data");
    context->IBM_texture_mirrored_repeat = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_IBM_texture_mirrored_repeat");
    context->IBM_vertex_array_lists = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_IBM_vertex_array_lists");
    context->INGR_blend_func_separate = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INGR_blend_func_separate");
    context->INGR_color_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INGR_color_clamp");
    context->INGR_interlace_read = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INGR_interlace_read");
    context->INTEL_blackhole_render = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INTEL_blackhole_render");
    context->INTEL_conservative_rasterization = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INTEL_conservative_rasterization");
    context->INTEL_fragment_shader_ordering = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INTEL_fragment_shader_ordering");
    context->INTEL_framebuffer_CMAA = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INTEL_framebuffer_CMAA");
    context->INTEL_map_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INTEL_map_texture");
    context->INTEL_parallel_arrays = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INTEL_parallel_arrays");
    context->INTEL_performance_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_INTEL_performance_query");
    context->KHR_blend_equation_advanced = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_blend_equation_advanced");
    context->KHR_blend_equation_advanced_coherent = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_blend_equation_advanced_coherent");
    context->KHR_context_flush_control = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_context_flush_control");
    context->KHR_debug = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_debug");
    context->KHR_no_error = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_no_error");
    context->KHR_parallel_shader_compile = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_parallel_shader_compile");
    context->KHR_robust_buffer_access_behavior = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_robust_buffer_access_behavior");
    context->KHR_robustness = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_robustness");
    context->KHR_shader_subgroup = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_shader_subgroup");
    context->KHR_texture_compression_astc_hdr = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_texture_compression_astc_hdr");
    context->KHR_texture_compression_astc_ldr = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_texture_compression_astc_ldr");
    context->KHR_texture_compression_astc_sliced_3d = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_KHR_texture_compression_astc_sliced_3d");
    context->MESAX_texture_stack = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESAX_texture_stack");
    context->MESA_framebuffer_flip_x = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_framebuffer_flip_x");
    context->MESA_framebuffer_flip_y = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_framebuffer_flip_y");
    context->MESA_framebuffer_swap_xy = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_framebuffer_swap_xy");
    context->MESA_pack_invert = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_pack_invert");
    context->MESA_program_binary_formats = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_program_binary_formats");
    context->MESA_resize_buffers = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_resize_buffers");
    context->MESA_shader_integer_functions = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_shader_integer_functions");
    context->MESA_tile_raster_order = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_tile_raster_order");
    context->MESA_window_pos = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_window_pos");
    context->MESA_ycbcr_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_MESA_ycbcr_texture");
    context->NVX_blend_equation_advanced_multi_draw_buffers = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NVX_blend_equation_advanced_multi_draw_buffers");
    context->NVX_conditional_render = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NVX_conditional_render");
    context->NVX_gpu_memory_info = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NVX_gpu_memory_info");
    context->NVX_gpu_multicast2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NVX_gpu_multicast2");
    context->NVX_linked_gpu_multicast = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NVX_linked_gpu_multicast");
    context->NVX_progress_fence = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NVX_progress_fence");
    context->NV_alpha_to_coverage_dither_control = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_alpha_to_coverage_dither_control");
    context->NV_bindless_multi_draw_indirect = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_bindless_multi_draw_indirect");
    context->NV_bindless_multi_draw_indirect_count = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_bindless_multi_draw_indirect_count");
    context->NV_bindless_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_bindless_texture");
    context->NV_blend_equation_advanced = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_blend_equation_advanced");
    context->NV_blend_equation_advanced_coherent = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_blend_equation_advanced_coherent");
    context->NV_blend_minmax_factor = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_blend_minmax_factor");
    context->NV_blend_square = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_blend_square");
    context->NV_clip_space_w_scaling = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_clip_space_w_scaling");
    context->NV_command_list = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_command_list");
    context->NV_compute_program5 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_compute_program5");
    context->NV_compute_shader_derivatives = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_compute_shader_derivatives");
    context->NV_conditional_render = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_conditional_render");
    context->NV_conservative_raster = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_conservative_raster");
    context->NV_conservative_raster_dilate = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_conservative_raster_dilate");
    context->NV_conservative_raster_pre_snap = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_conservative_raster_pre_snap");
    context->NV_conservative_raster_pre_snap_triangles = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_conservative_raster_pre_snap_triangles");
    context->NV_conservative_raster_underestimation = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_conservative_raster_underestimation");
    context->NV_copy_depth_to_color = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_copy_depth_to_color");
    context->NV_copy_image = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_copy_image");
    context->NV_deep_texture3D = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_deep_texture3D");
    context->NV_depth_buffer_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_depth_buffer_float");
    context->NV_depth_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_depth_clamp");
    context->NV_draw_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_draw_texture");
    context->NV_draw_vulkan_image = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_draw_vulkan_image");
    context->NV_evaluators = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_evaluators");
    context->NV_explicit_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_explicit_multisample");
    context->NV_fence = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fence");
    context->NV_fill_rectangle = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fill_rectangle");
    context->NV_float_buffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_float_buffer");
    context->NV_fog_distance = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fog_distance");
    context->NV_fragment_coverage_to_color = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fragment_coverage_to_color");
    context->NV_fragment_program = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fragment_program");
    context->NV_fragment_program2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fragment_program2");
    context->NV_fragment_program4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fragment_program4");
    context->NV_fragment_program_option = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fragment_program_option");
    context->NV_fragment_shader_barycentric = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fragment_shader_barycentric");
    context->NV_fragment_shader_interlock = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_fragment_shader_interlock");
    context->NV_framebuffer_mixed_samples = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_framebuffer_mixed_samples");
    context->NV_framebuffer_multisample_coverage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_framebuffer_multisample_coverage");
    context->NV_geometry_program4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_geometry_program4");
    context->NV_geometry_shader4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_geometry_shader4");
    context->NV_geometry_shader_passthrough = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_geometry_shader_passthrough");
    context->NV_gpu_multicast = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_gpu_multicast");
    context->NV_gpu_program4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_gpu_program4");
    context->NV_gpu_program5 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_gpu_program5");
    context->NV_gpu_program5_mem_extended = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_gpu_program5_mem_extended");
    context->NV_gpu_shader5 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_gpu_shader5");
    context->NV_half_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_half_float");
    context->NV_internalformat_sample_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_internalformat_sample_query");
    context->NV_light_max_exponent = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_light_max_exponent");
    context->NV_memory_attachment = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_memory_attachment");
    context->NV_memory_object_sparse = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_memory_object_sparse");
    context->NV_mesh_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_mesh_shader");
    context->NV_multisample_coverage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_multisample_coverage");
    context->NV_multisample_filter_hint = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_multisample_filter_hint");
    context->NV_occlusion_query = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_occlusion_query");
    context->NV_packed_depth_stencil = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_packed_depth_stencil");
    context->NV_parameter_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_parameter_buffer_object");
    context->NV_parameter_buffer_object2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_parameter_buffer_object2");
    context->NV_path_rendering = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_path_rendering");
    context->NV_path_rendering_shared_edge = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_path_rendering_shared_edge");
    context->NV_pixel_data_range = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_pixel_data_range");
    context->NV_point_sprite = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_point_sprite");
    context->NV_present_video = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_present_video");
    context->NV_primitive_restart = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_primitive_restart");
    context->NV_primitive_shading_rate = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_primitive_shading_rate");
    context->NV_query_resource = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_query_resource");
    context->NV_query_resource_tag = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_query_resource_tag");
    context->NV_register_combiners = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_register_combiners");
    context->NV_register_combiners2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_register_combiners2");
    context->NV_representative_fragment_test = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_representative_fragment_test");
    context->NV_robustness_video_memory_purge = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_robustness_video_memory_purge");
    context->NV_sample_locations = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_sample_locations");
    context->NV_sample_mask_override_coverage = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_sample_mask_override_coverage");
    context->NV_scissor_exclusive = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_scissor_exclusive");
    context->NV_shader_atomic_counters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_atomic_counters");
    context->NV_shader_atomic_float = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_atomic_float");
    context->NV_shader_atomic_float64 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_atomic_float64");
    context->NV_shader_atomic_fp16_vector = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_atomic_fp16_vector");
    context->NV_shader_atomic_int64 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_atomic_int64");
    context->NV_shader_buffer_load = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_buffer_load");
    context->NV_shader_buffer_store = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_buffer_store");
    context->NV_shader_storage_buffer_object = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_storage_buffer_object");
    context->NV_shader_subgroup_partitioned = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_subgroup_partitioned");
    context->NV_shader_texture_footprint = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_texture_footprint");
    context->NV_shader_thread_group = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_thread_group");
    context->NV_shader_thread_shuffle = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shader_thread_shuffle");
    context->NV_shading_rate_image = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_shading_rate_image");
    context->NV_stereo_view_rendering = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_stereo_view_rendering");
    context->NV_tessellation_program5 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_tessellation_program5");
    context->NV_texgen_emboss = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texgen_emboss");
    context->NV_texgen_reflection = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texgen_reflection");
    context->NV_texture_barrier = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_barrier");
    context->NV_texture_compression_vtc = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_compression_vtc");
    context->NV_texture_env_combine4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_env_combine4");
    context->NV_texture_expand_normal = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_expand_normal");
    context->NV_texture_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_multisample");
    context->NV_texture_rectangle = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_rectangle");
    context->NV_texture_rectangle_compressed = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_rectangle_compressed");
    context->NV_texture_shader = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_shader");
    context->NV_texture_shader2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_shader2");
    context->NV_texture_shader3 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_texture_shader3");
    context->NV_timeline_semaphore = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_timeline_semaphore");
    context->NV_transform_feedback = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_transform_feedback");
    context->NV_transform_feedback2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_transform_feedback2");
    context->NV_uniform_buffer_unified_memory = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_uniform_buffer_unified_memory");
    context->NV_vdpau_interop = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vdpau_interop");
    context->NV_vdpau_interop2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vdpau_interop2");
    context->NV_vertex_array_range = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_array_range");
    context->NV_vertex_array_range2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_array_range2");
    context->NV_vertex_attrib_integer_64bit = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_attrib_integer_64bit");
    context->NV_vertex_buffer_unified_memory = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_buffer_unified_memory");
    context->NV_vertex_program = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_program");
    context->NV_vertex_program1_1 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_program1_1");
    context->NV_vertex_program2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_program2");
    context->NV_vertex_program2_option = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_program2_option");
    context->NV_vertex_program3 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_program3");
    context->NV_vertex_program4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_vertex_program4");
    context->NV_video_capture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_video_capture");
    context->NV_viewport_array2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_viewport_array2");
    context->NV_viewport_swizzle = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_NV_viewport_swizzle");
    context->OES_byte_coordinates = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OES_byte_coordinates");
    context->OES_compressed_paletted_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OES_compressed_paletted_texture");
    context->OES_fixed_point = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OES_fixed_point");
    context->OES_query_matrix = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OES_query_matrix");
    context->OES_read_format = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OES_read_format");
    context->OES_single_precision = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OES_single_precision");
    context->OML_interlace = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OML_interlace");
    context->OML_resample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OML_resample");
    context->OML_subsample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OML_subsample");
    context->OVR_multiview = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OVR_multiview");
    context->OVR_multiview2 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_OVR_multiview2");
    context->PGI_misc_hints = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_PGI_misc_hints");
    context->PGI_vertex_hints = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_PGI_vertex_hints");
    context->REND_screen_coordinates = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_REND_screen_coordinates");
    context->S3_s3tc = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_S3_s3tc");
    context->SGIS_detail_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_detail_texture");
    context->SGIS_fog_function = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_fog_function");
    context->SGIS_generate_mipmap = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_generate_mipmap");
    context->SGIS_multisample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_multisample");
    context->SGIS_pixel_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_pixel_texture");
    context->SGIS_point_line_texgen = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_point_line_texgen");
    context->SGIS_point_parameters = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_point_parameters");
    context->SGIS_sharpen_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_sharpen_texture");
    context->SGIS_texture4D = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_texture4D");
    context->SGIS_texture_border_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_texture_border_clamp");
    context->SGIS_texture_color_mask = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_texture_color_mask");
    context->SGIS_texture_edge_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_texture_edge_clamp");
    context->SGIS_texture_filter4 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_texture_filter4");
    context->SGIS_texture_lod = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_texture_lod");
    context->SGIS_texture_select = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIS_texture_select");
    context->SGIX_async = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_async");
    context->SGIX_async_histogram = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_async_histogram");
    context->SGIX_async_pixel = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_async_pixel");
    context->SGIX_blend_alpha_minmax = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_blend_alpha_minmax");
    context->SGIX_calligraphic_fragment = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_calligraphic_fragment");
    context->SGIX_clipmap = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_clipmap");
    context->SGIX_convolution_accuracy = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_convolution_accuracy");
    context->SGIX_depth_pass_instrument = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_depth_pass_instrument");
    context->SGIX_depth_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_depth_texture");
    context->SGIX_flush_raster = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_flush_raster");
    context->SGIX_fog_offset = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_fog_offset");
    context->SGIX_fragment_lighting = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_fragment_lighting");
    context->SGIX_framezoom = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_framezoom");
    context->SGIX_igloo_interface = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_igloo_interface");
    context->SGIX_instruments = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_instruments");
    context->SGIX_interlace = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_interlace");
    context->SGIX_ir_instrument1 = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_ir_instrument1");
    context->SGIX_list_priority = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_list_priority");
    context->SGIX_pixel_texture = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_pixel_texture");
    context->SGIX_pixel_tiles = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_pixel_tiles");
    context->SGIX_polynomial_ffd = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_polynomial_ffd");
    context->SGIX_reference_plane = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_reference_plane");
    context->SGIX_resample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_resample");
    context->SGIX_scalebias_hint = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_scalebias_hint");
    context->SGIX_shadow = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_shadow");
    context->SGIX_shadow_ambient = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_shadow_ambient");
    context->SGIX_sprite = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_sprite");
    context->SGIX_subsample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_subsample");
    context->SGIX_tag_sample_buffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_tag_sample_buffer");
    context->SGIX_texture_add_env = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_texture_add_env");
    context->SGIX_texture_coordinate_clamp = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_texture_coordinate_clamp");
    context->SGIX_texture_lod_bias = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_texture_lod_bias");
    context->SGIX_texture_multi_buffer = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_texture_multi_buffer");
    context->SGIX_texture_scale_bias = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_texture_scale_bias");
    context->SGIX_vertex_preclip = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_vertex_preclip");
    context->SGIX_ycrcb = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_ycrcb");
    context->SGIX_ycrcb_subsample = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_ycrcb_subsample");
    context->SGIX_ycrcba = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGIX_ycrcba");
    context->SGI_color_matrix = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGI_color_matrix");
    context->SGI_color_table = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGI_color_table");
    context->SGI_texture_color_table = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SGI_texture_color_table");
    context->SUNX_constant_data = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SUNX_constant_data");
    context->SUN_convolution_border_modes = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SUN_convolution_border_modes");
    context->SUN_global_alpha = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SUN_global_alpha");
    context->SUN_mesh_array = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SUN_mesh_array");
    context->SUN_slice_accum = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SUN_slice_accum");
    context->SUN_triangle_list = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SUN_triangle_list");
    context->SUN_vertex = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_SUN_vertex");
    context->WIN_phong_shading = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_WIN_phong_shading");
    context->WIN_specular_fog = glad_gl_has_extension(version, exts, num_exts_i, exts_i, "GL_WIN_specular_fog");

    glad_gl_free_extensions(exts_i, num_exts_i);

    return 1;
}

static int glad_gl_find_core_gl(GladGLContext *context) {
    int i;
    const char* version;
    const char* prefixes[] = {
        "OpenGL ES-CM ",
        "OpenGL ES-CL ",
        "OpenGL ES ",
        "OpenGL SC ",
        NULL
    };
    int major = 0;
    int minor = 0;
    version = (const char*) context->GetString(GL_VERSION);
    if (!version) return 0;
    for (i = 0;  prefixes[i];  i++) {
        const size_t length = strlen(prefixes[i]);
        if (strncmp(version, prefixes[i], length) == 0) {
            version += length;
            break;
        }
    }

    GLAD_IMPL_UTIL_SSCANF(version, "%d.%d", &major, &minor);

    context->VERSION_1_0 = (major == 1 && minor >= 0) || major > 1;
    context->VERSION_1_1 = (major == 1 && minor >= 1) || major > 1;
    context->VERSION_1_2 = (major == 1 && minor >= 2) || major > 1;
    context->VERSION_1_3 = (major == 1 && minor >= 3) || major > 1;
    context->VERSION_1_4 = (major == 1 && minor >= 4) || major > 1;
    context->VERSION_1_5 = (major == 1 && minor >= 5) || major > 1;
    context->VERSION_2_0 = (major == 2 && minor >= 0) || major > 2;
    context->VERSION_2_1 = (major == 2 && minor >= 1) || major > 2;
    context->VERSION_3_0 = (major == 3 && minor >= 0) || major > 3;
    context->VERSION_3_1 = (major == 3 && minor >= 1) || major > 3;
    context->VERSION_3_2 = (major == 3 && minor >= 2) || major > 3;
    context->VERSION_3_3 = (major == 3 && minor >= 3) || major > 3;
    context->VERSION_4_0 = (major == 4 && minor >= 0) || major > 4;
    context->VERSION_4_1 = (major == 4 && minor >= 1) || major > 4;
    context->VERSION_4_2 = (major == 4 && minor >= 2) || major > 4;
    context->VERSION_4_3 = (major == 4 && minor >= 3) || major > 4;
    context->VERSION_4_4 = (major == 4 && minor >= 4) || major > 4;
    context->VERSION_4_5 = (major == 4 && minor >= 5) || major > 4;
    context->VERSION_4_6 = (major == 4 && minor >= 6) || major > 4;

    return GLAD_MAKE_VERSION(major, minor);
}

int gladLoadGLContextUserPtr(GladGLContext *context, GLADuserptrloadfunc load, void *userptr) {
    int version;

    context->GetString = (PFNGLGETSTRINGPROC) load(userptr, "glGetString");
    if(context->GetString == NULL) return 0;
    if(context->GetString(GL_VERSION) == NULL) return 0;
    version = glad_gl_find_core_gl(context);

    glad_gl_load_GL_VERSION_1_0(context, load, userptr);
    glad_gl_load_GL_VERSION_1_1(context, load, userptr);
    glad_gl_load_GL_VERSION_1_2(context, load, userptr);
    glad_gl_load_GL_VERSION_1_3(context, load, userptr);
    glad_gl_load_GL_VERSION_1_4(context, load, userptr);
    glad_gl_load_GL_VERSION_1_5(context, load, userptr);
    glad_gl_load_GL_VERSION_2_0(context, load, userptr);
    glad_gl_load_GL_VERSION_2_1(context, load, userptr);
    glad_gl_load_GL_VERSION_3_0(context, load, userptr);
    glad_gl_load_GL_VERSION_3_1(context, load, userptr);
    glad_gl_load_GL_VERSION_3_2(context, load, userptr);
    glad_gl_load_GL_VERSION_3_3(context, load, userptr);
    glad_gl_load_GL_VERSION_4_0(context, load, userptr);
    glad_gl_load_GL_VERSION_4_1(context, load, userptr);
    glad_gl_load_GL_VERSION_4_2(context, load, userptr);
    glad_gl_load_GL_VERSION_4_3(context, load, userptr);
    glad_gl_load_GL_VERSION_4_4(context, load, userptr);
    glad_gl_load_GL_VERSION_4_5(context, load, userptr);
    glad_gl_load_GL_VERSION_4_6(context, load, userptr);

    if (!glad_gl_find_extensions_gl(context, version)) return 0;
    glad_gl_load_GL_3DFX_tbuffer(context, load, userptr);
    glad_gl_load_GL_AMD_debug_output(context, load, userptr);
    glad_gl_load_GL_AMD_draw_buffers_blend(context, load, userptr);
    glad_gl_load_GL_AMD_framebuffer_multisample_advanced(context, load, userptr);
    glad_gl_load_GL_AMD_framebuffer_sample_positions(context, load, userptr);
    glad_gl_load_GL_AMD_gpu_shader_int64(context, load, userptr);
    glad_gl_load_GL_AMD_interleaved_elements(context, load, userptr);
    glad_gl_load_GL_AMD_multi_draw_indirect(context, load, userptr);
    glad_gl_load_GL_AMD_name_gen_delete(context, load, userptr);
    glad_gl_load_GL_AMD_occlusion_query_event(context, load, userptr);
    glad_gl_load_GL_AMD_performance_monitor(context, load, userptr);
    glad_gl_load_GL_AMD_sample_positions(context, load, userptr);
    glad_gl_load_GL_AMD_sparse_texture(context, load, userptr);
    glad_gl_load_GL_AMD_stencil_operation_extended(context, load, userptr);
    glad_gl_load_GL_AMD_vertex_shader_tessellator(context, load, userptr);
    glad_gl_load_GL_APPLE_element_array(context, load, userptr);
    glad_gl_load_GL_APPLE_fence(context, load, userptr);
    glad_gl_load_GL_APPLE_flush_buffer_range(context, load, userptr);
    glad_gl_load_GL_APPLE_object_purgeable(context, load, userptr);
    glad_gl_load_GL_APPLE_texture_range(context, load, userptr);
    glad_gl_load_GL_APPLE_vertex_array_object(context, load, userptr);
    glad_gl_load_GL_APPLE_vertex_array_range(context, load, userptr);
    glad_gl_load_GL_APPLE_vertex_program_evaluators(context, load, userptr);
    glad_gl_load_GL_ARB_ES2_compatibility(context, load, userptr);
    glad_gl_load_GL_ARB_ES3_1_compatibility(context, load, userptr);
    glad_gl_load_GL_ARB_ES3_2_compatibility(context, load, userptr);
    glad_gl_load_GL_ARB_base_instance(context, load, userptr);
    glad_gl_load_GL_ARB_bindless_texture(context, load, userptr);
    glad_gl_load_GL_ARB_blend_func_extended(context, load, userptr);
    glad_gl_load_GL_ARB_buffer_storage(context, load, userptr);
    glad_gl_load_GL_ARB_cl_event(context, load, userptr);
    glad_gl_load_GL_ARB_clear_buffer_object(context, load, userptr);
    glad_gl_load_GL_ARB_clear_texture(context, load, userptr);
    glad_gl_load_GL_ARB_clip_control(context, load, userptr);
    glad_gl_load_GL_ARB_color_buffer_float(context, load, userptr);
    glad_gl_load_GL_ARB_compute_shader(context, load, userptr);
    glad_gl_load_GL_ARB_compute_variable_group_size(context, load, userptr);
    glad_gl_load_GL_ARB_copy_buffer(context, load, userptr);
    glad_gl_load_GL_ARB_copy_image(context, load, userptr);
    glad_gl_load_GL_ARB_debug_output(context, load, userptr);
    glad_gl_load_GL_ARB_direct_state_access(context, load, userptr);
    glad_gl_load_GL_ARB_draw_buffers(context, load, userptr);
    glad_gl_load_GL_ARB_draw_buffers_blend(context, load, userptr);
    glad_gl_load_GL_ARB_draw_elements_base_vertex(context, load, userptr);
    glad_gl_load_GL_ARB_draw_indirect(context, load, userptr);
    glad_gl_load_GL_ARB_draw_instanced(context, load, userptr);
    glad_gl_load_GL_ARB_fragment_program(context, load, userptr);
    glad_gl_load_GL_ARB_framebuffer_no_attachments(context, load, userptr);
    glad_gl_load_GL_ARB_framebuffer_object(context, load, userptr);
    glad_gl_load_GL_ARB_geometry_shader4(context, load, userptr);
    glad_gl_load_GL_ARB_get_program_binary(context, load, userptr);
    glad_gl_load_GL_ARB_get_texture_sub_image(context, load, userptr);
    glad_gl_load_GL_ARB_gl_spirv(context, load, userptr);
    glad_gl_load_GL_ARB_gpu_shader_fp64(context, load, userptr);
    glad_gl_load_GL_ARB_gpu_shader_int64(context, load, userptr);
    glad_gl_load_GL_ARB_imaging(context, load, userptr);
    glad_gl_load_GL_ARB_indirect_parameters(context, load, userptr);
    glad_gl_load_GL_ARB_instanced_arrays(context, load, userptr);
    glad_gl_load_GL_ARB_internalformat_query(context, load, userptr);
    glad_gl_load_GL_ARB_internalformat_query2(context, load, userptr);
    glad_gl_load_GL_ARB_invalidate_subdata(context, load, userptr);
    glad_gl_load_GL_ARB_map_buffer_range(context, load, userptr);
    glad_gl_load_GL_ARB_matrix_palette(context, load, userptr);
    glad_gl_load_GL_ARB_multi_bind(context, load, userptr);
    glad_gl_load_GL_ARB_multi_draw_indirect(context, load, userptr);
    glad_gl_load_GL_ARB_multisample(context, load, userptr);
    glad_gl_load_GL_ARB_multitexture(context, load, userptr);
    glad_gl_load_GL_ARB_occlusion_query(context, load, userptr);
    glad_gl_load_GL_ARB_parallel_shader_compile(context, load, userptr);
    glad_gl_load_GL_ARB_point_parameters(context, load, userptr);
    glad_gl_load_GL_ARB_polygon_offset_clamp(context, load, userptr);
    glad_gl_load_GL_ARB_program_interface_query(context, load, userptr);
    glad_gl_load_GL_ARB_provoking_vertex(context, load, userptr);
    glad_gl_load_GL_ARB_robustness(context, load, userptr);
    glad_gl_load_GL_ARB_sample_locations(context, load, userptr);
    glad_gl_load_GL_ARB_sample_shading(context, load, userptr);
    glad_gl_load_GL_ARB_sampler_objects(context, load, userptr);
    glad_gl_load_GL_ARB_separate_shader_objects(context, load, userptr);
    glad_gl_load_GL_ARB_shader_atomic_counters(context, load, userptr);
    glad_gl_load_GL_ARB_shader_image_load_store(context, load, userptr);
    glad_gl_load_GL_ARB_shader_objects(context, load, userptr);
    glad_gl_load_GL_ARB_shader_storage_buffer_object(context, load, userptr);
    glad_gl_load_GL_ARB_shader_subroutine(context, load, userptr);
    glad_gl_load_GL_ARB_shading_language_include(context, load, userptr);
    glad_gl_load_GL_ARB_sparse_buffer(context, load, userptr);
    glad_gl_load_GL_ARB_sparse_texture(context, load, userptr);
    glad_gl_load_GL_ARB_sync(context, load, userptr);
    glad_gl_load_GL_ARB_tessellation_shader(context, load, userptr);
    glad_gl_load_GL_ARB_texture_barrier(context, load, userptr);
    glad_gl_load_GL_ARB_texture_buffer_object(context, load, userptr);
    glad_gl_load_GL_ARB_texture_buffer_range(context, load, userptr);
    glad_gl_load_GL_ARB_texture_compression(context, load, userptr);
    glad_gl_load_GL_ARB_texture_multisample(context, load, userptr);
    glad_gl_load_GL_ARB_texture_storage(context, load, userptr);
    glad_gl_load_GL_ARB_texture_storage_multisample(context, load, userptr);
    glad_gl_load_GL_ARB_texture_view(context, load, userptr);
    glad_gl_load_GL_ARB_timer_query(context, load, userptr);
    glad_gl_load_GL_ARB_transform_feedback2(context, load, userptr);
    glad_gl_load_GL_ARB_transform_feedback3(context, load, userptr);
    glad_gl_load_GL_ARB_transform_feedback_instanced(context, load, userptr);
    glad_gl_load_GL_ARB_transpose_matrix(context, load, userptr);
    glad_gl_load_GL_ARB_uniform_buffer_object(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_array_object(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_attrib_64bit(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_attrib_binding(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_blend(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_buffer_object(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_program(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_shader(context, load, userptr);
    glad_gl_load_GL_ARB_vertex_type_2_10_10_10_rev(context, load, userptr);
    glad_gl_load_GL_ARB_viewport_array(context, load, userptr);
    glad_gl_load_GL_ARB_window_pos(context, load, userptr);
    glad_gl_load_GL_ATI_draw_buffers(context, load, userptr);
    glad_gl_load_GL_ATI_element_array(context, load, userptr);
    glad_gl_load_GL_ATI_envmap_bumpmap(context, load, userptr);
    glad_gl_load_GL_ATI_fragment_shader(context, load, userptr);
    glad_gl_load_GL_ATI_map_object_buffer(context, load, userptr);
    glad_gl_load_GL_ATI_pn_triangles(context, load, userptr);
    glad_gl_load_GL_ATI_separate_stencil(context, load, userptr);
    glad_gl_load_GL_ATI_vertex_array_object(context, load, userptr);
    glad_gl_load_GL_ATI_vertex_attrib_array_object(context, load, userptr);
    glad_gl_load_GL_ATI_vertex_streams(context, load, userptr);
    glad_gl_load_GL_EXT_EGL_image_storage(context, load, userptr);
    glad_gl_load_GL_EXT_bindable_uniform(context, load, userptr);
    glad_gl_load_GL_EXT_blend_color(context, load, userptr);
    glad_gl_load_GL_EXT_blend_equation_separate(context, load, userptr);
    glad_gl_load_GL_EXT_blend_func_separate(context, load, userptr);
    glad_gl_load_GL_EXT_blend_minmax(context, load, userptr);
    glad_gl_load_GL_EXT_color_subtable(context, load, userptr);
    glad_gl_load_GL_EXT_compiled_vertex_array(context, load, userptr);
    glad_gl_load_GL_EXT_convolution(context, load, userptr);
    glad_gl_load_GL_EXT_coordinate_frame(context, load, userptr);
    glad_gl_load_GL_EXT_copy_texture(context, load, userptr);
    glad_gl_load_GL_EXT_cull_vertex(context, load, userptr);
    glad_gl_load_GL_EXT_debug_label(context, load, userptr);
    glad_gl_load_GL_EXT_debug_marker(context, load, userptr);
    glad_gl_load_GL_EXT_depth_bounds_test(context, load, userptr);
    glad_gl_load_GL_EXT_direct_state_access(context, load, userptr);
    glad_gl_load_GL_EXT_draw_buffers2(context, load, userptr);
    glad_gl_load_GL_EXT_draw_instanced(context, load, userptr);
    glad_gl_load_GL_EXT_draw_range_elements(context, load, userptr);
    glad_gl_load_GL_EXT_external_buffer(context, load, userptr);
    glad_gl_load_GL_EXT_fog_coord(context, load, userptr);
    glad_gl_load_GL_EXT_framebuffer_blit(context, load, userptr);
    glad_gl_load_GL_EXT_framebuffer_multisample(context, load, userptr);
    glad_gl_load_GL_EXT_framebuffer_object(context, load, userptr);
    glad_gl_load_GL_EXT_geometry_shader4(context, load, userptr);
    glad_gl_load_GL_EXT_gpu_program_parameters(context, load, userptr);
    glad_gl_load_GL_EXT_gpu_shader4(context, load, userptr);
    glad_gl_load_GL_EXT_histogram(context, load, userptr);
    glad_gl_load_GL_EXT_index_func(context, load, userptr);
    glad_gl_load_GL_EXT_index_material(context, load, userptr);
    glad_gl_load_GL_EXT_light_texture(context, load, userptr);
    glad_gl_load_GL_EXT_memory_object(context, load, userptr);
    glad_gl_load_GL_EXT_memory_object_fd(context, load, userptr);
    glad_gl_load_GL_EXT_memory_object_win32(context, load, userptr);
    glad_gl_load_GL_EXT_multi_draw_arrays(context, load, userptr);
    glad_gl_load_GL_EXT_multisample(context, load, userptr);
    glad_gl_load_GL_EXT_paletted_texture(context, load, userptr);
    glad_gl_load_GL_EXT_pixel_transform(context, load, userptr);
    glad_gl_load_GL_EXT_point_parameters(context, load, userptr);
    glad_gl_load_GL_EXT_polygon_offset(context, load, userptr);
    glad_gl_load_GL_EXT_polygon_offset_clamp(context, load, userptr);
    glad_gl_load_GL_EXT_provoking_vertex(context, load, userptr);
    glad_gl_load_GL_EXT_raster_multisample(context, load, userptr);
    glad_gl_load_GL_EXT_secondary_color(context, load, userptr);
    glad_gl_load_GL_EXT_semaphore(context, load, userptr);
    glad_gl_load_GL_EXT_semaphore_fd(context, load, userptr);
    glad_gl_load_GL_EXT_semaphore_win32(context, load, userptr);
    glad_gl_load_GL_EXT_separate_shader_objects(context, load, userptr);
    glad_gl_load_GL_EXT_shader_framebuffer_fetch_non_coherent(context, load, userptr);
    glad_gl_load_GL_EXT_shader_image_load_store(context, load, userptr);
    glad_gl_load_GL_EXT_stencil_clear_tag(context, load, userptr);
    glad_gl_load_GL_EXT_stencil_two_side(context, load, userptr);
    glad_gl_load_GL_EXT_subtexture(context, load, userptr);
    glad_gl_load_GL_EXT_texture3D(context, load, userptr);
    glad_gl_load_GL_EXT_texture_array(context, load, userptr);
    glad_gl_load_GL_EXT_texture_buffer_object(context, load, userptr);
    glad_gl_load_GL_EXT_texture_integer(context, load, userptr);
    glad_gl_load_GL_EXT_texture_object(context, load, userptr);
    glad_gl_load_GL_EXT_texture_perturb_normal(context, load, userptr);
    glad_gl_load_GL_EXT_texture_storage(context, load, userptr);
    glad_gl_load_GL_EXT_timer_query(context, load, userptr);
    glad_gl_load_GL_EXT_transform_feedback(context, load, userptr);
    glad_gl_load_GL_EXT_vertex_array(context, load, userptr);
    glad_gl_load_GL_EXT_vertex_attrib_64bit(context, load, userptr);
    glad_gl_load_GL_EXT_vertex_shader(context, load, userptr);
    glad_gl_load_GL_EXT_vertex_weighting(context, load, userptr);
    glad_gl_load_GL_EXT_win32_keyed_mutex(context, load, userptr);
    glad_gl_load_GL_EXT_window_rectangles(context, load, userptr);
    glad_gl_load_GL_EXT_x11_sync_object(context, load, userptr);
    glad_gl_load_GL_GREMEDY_frame_terminator(context, load, userptr);
    glad_gl_load_GL_GREMEDY_string_marker(context, load, userptr);
    glad_gl_load_GL_HP_image_transform(context, load, userptr);
    glad_gl_load_GL_IBM_multimode_draw_arrays(context, load, userptr);
    glad_gl_load_GL_IBM_static_data(context, load, userptr);
    glad_gl_load_GL_IBM_vertex_array_lists(context, load, userptr);
    glad_gl_load_GL_INGR_blend_func_separate(context, load, userptr);
    glad_gl_load_GL_INTEL_framebuffer_CMAA(context, load, userptr);
    glad_gl_load_GL_INTEL_map_texture(context, load, userptr);
    glad_gl_load_GL_INTEL_parallel_arrays(context, load, userptr);
    glad_gl_load_GL_INTEL_performance_query(context, load, userptr);
    glad_gl_load_GL_KHR_blend_equation_advanced(context, load, userptr);
    glad_gl_load_GL_KHR_debug(context, load, userptr);
    glad_gl_load_GL_KHR_parallel_shader_compile(context, load, userptr);
    glad_gl_load_GL_KHR_robustness(context, load, userptr);
    glad_gl_load_GL_MESA_framebuffer_flip_y(context, load, userptr);
    glad_gl_load_GL_MESA_resize_buffers(context, load, userptr);
    glad_gl_load_GL_MESA_window_pos(context, load, userptr);
    glad_gl_load_GL_NVX_conditional_render(context, load, userptr);
    glad_gl_load_GL_NVX_gpu_multicast2(context, load, userptr);
    glad_gl_load_GL_NVX_linked_gpu_multicast(context, load, userptr);
    glad_gl_load_GL_NVX_progress_fence(context, load, userptr);
    glad_gl_load_GL_NV_alpha_to_coverage_dither_control(context, load, userptr);
    glad_gl_load_GL_NV_bindless_multi_draw_indirect(context, load, userptr);
    glad_gl_load_GL_NV_bindless_multi_draw_indirect_count(context, load, userptr);
    glad_gl_load_GL_NV_bindless_texture(context, load, userptr);
    glad_gl_load_GL_NV_blend_equation_advanced(context, load, userptr);
    glad_gl_load_GL_NV_clip_space_w_scaling(context, load, userptr);
    glad_gl_load_GL_NV_command_list(context, load, userptr);
    glad_gl_load_GL_NV_conditional_render(context, load, userptr);
    glad_gl_load_GL_NV_conservative_raster(context, load, userptr);
    glad_gl_load_GL_NV_conservative_raster_dilate(context, load, userptr);
    glad_gl_load_GL_NV_conservative_raster_pre_snap_triangles(context, load, userptr);
    glad_gl_load_GL_NV_copy_image(context, load, userptr);
    glad_gl_load_GL_NV_depth_buffer_float(context, load, userptr);
    glad_gl_load_GL_NV_draw_texture(context, load, userptr);
    glad_gl_load_GL_NV_draw_vulkan_image(context, load, userptr);
    glad_gl_load_GL_NV_evaluators(context, load, userptr);
    glad_gl_load_GL_NV_explicit_multisample(context, load, userptr);
    glad_gl_load_GL_NV_fence(context, load, userptr);
    glad_gl_load_GL_NV_fragment_coverage_to_color(context, load, userptr);
    glad_gl_load_GL_NV_fragment_program(context, load, userptr);
    glad_gl_load_GL_NV_framebuffer_mixed_samples(context, load, userptr);
    glad_gl_load_GL_NV_framebuffer_multisample_coverage(context, load, userptr);
    glad_gl_load_GL_NV_geometry_program4(context, load, userptr);
    glad_gl_load_GL_NV_gpu_multicast(context, load, userptr);
    glad_gl_load_GL_NV_gpu_program4(context, load, userptr);
    glad_gl_load_GL_NV_gpu_program5(context, load, userptr);
    glad_gl_load_GL_NV_gpu_shader5(context, load, userptr);
    glad_gl_load_GL_NV_half_float(context, load, userptr);
    glad_gl_load_GL_NV_internalformat_sample_query(context, load, userptr);
    glad_gl_load_GL_NV_memory_attachment(context, load, userptr);
    glad_gl_load_GL_NV_memory_object_sparse(context, load, userptr);
    glad_gl_load_GL_NV_mesh_shader(context, load, userptr);
    glad_gl_load_GL_NV_occlusion_query(context, load, userptr);
    glad_gl_load_GL_NV_parameter_buffer_object(context, load, userptr);
    glad_gl_load_GL_NV_path_rendering(context, load, userptr);
    glad_gl_load_GL_NV_pixel_data_range(context, load, userptr);
    glad_gl_load_GL_NV_point_sprite(context, load, userptr);
    glad_gl_load_GL_NV_present_video(context, load, userptr);
    glad_gl_load_GL_NV_primitive_restart(context, load, userptr);
    glad_gl_load_GL_NV_query_resource(context, load, userptr);
    glad_gl_load_GL_NV_query_resource_tag(context, load, userptr);
    glad_gl_load_GL_NV_register_combiners(context, load, userptr);
    glad_gl_load_GL_NV_register_combiners2(context, load, userptr);
    glad_gl_load_GL_NV_sample_locations(context, load, userptr);
    glad_gl_load_GL_NV_scissor_exclusive(context, load, userptr);
    glad_gl_load_GL_NV_shader_buffer_load(context, load, userptr);
    glad_gl_load_GL_NV_shading_rate_image(context, load, userptr);
    glad_gl_load_GL_NV_texture_barrier(context, load, userptr);
    glad_gl_load_GL_NV_texture_multisample(context, load, userptr);
    glad_gl_load_GL_NV_timeline_semaphore(context, load, userptr);
    glad_gl_load_GL_NV_transform_feedback(context, load, userptr);
    glad_gl_load_GL_NV_transform_feedback2(context, load, userptr);
    glad_gl_load_GL_NV_vdpau_interop(context, load, userptr);
    glad_gl_load_GL_NV_vdpau_interop2(context, load, userptr);
    glad_gl_load_GL_NV_vertex_array_range(context, load, userptr);
    glad_gl_load_GL_NV_vertex_attrib_integer_64bit(context, load, userptr);
    glad_gl_load_GL_NV_vertex_buffer_unified_memory(context, load, userptr);
    glad_gl_load_GL_NV_vertex_program(context, load, userptr);
    glad_gl_load_GL_NV_vertex_program4(context, load, userptr);
    glad_gl_load_GL_NV_video_capture(context, load, userptr);
    glad_gl_load_GL_NV_viewport_swizzle(context, load, userptr);
    glad_gl_load_GL_OES_byte_coordinates(context, load, userptr);
    glad_gl_load_GL_OES_fixed_point(context, load, userptr);
    glad_gl_load_GL_OES_query_matrix(context, load, userptr);
    glad_gl_load_GL_OES_single_precision(context, load, userptr);
    glad_gl_load_GL_OVR_multiview(context, load, userptr);
    glad_gl_load_GL_PGI_misc_hints(context, load, userptr);
    glad_gl_load_GL_SGIS_detail_texture(context, load, userptr);
    glad_gl_load_GL_SGIS_fog_function(context, load, userptr);
    glad_gl_load_GL_SGIS_multisample(context, load, userptr);
    glad_gl_load_GL_SGIS_pixel_texture(context, load, userptr);
    glad_gl_load_GL_SGIS_point_parameters(context, load, userptr);
    glad_gl_load_GL_SGIS_sharpen_texture(context, load, userptr);
    glad_gl_load_GL_SGIS_texture4D(context, load, userptr);
    glad_gl_load_GL_SGIS_texture_color_mask(context, load, userptr);
    glad_gl_load_GL_SGIS_texture_filter4(context, load, userptr);
    glad_gl_load_GL_SGIX_async(context, load, userptr);
    glad_gl_load_GL_SGIX_flush_raster(context, load, userptr);
    glad_gl_load_GL_SGIX_fragment_lighting(context, load, userptr);
    glad_gl_load_GL_SGIX_framezoom(context, load, userptr);
    glad_gl_load_GL_SGIX_igloo_interface(context, load, userptr);
    glad_gl_load_GL_SGIX_instruments(context, load, userptr);
    glad_gl_load_GL_SGIX_list_priority(context, load, userptr);
    glad_gl_load_GL_SGIX_pixel_texture(context, load, userptr);
    glad_gl_load_GL_SGIX_polynomial_ffd(context, load, userptr);
    glad_gl_load_GL_SGIX_reference_plane(context, load, userptr);
    glad_gl_load_GL_SGIX_sprite(context, load, userptr);
    glad_gl_load_GL_SGIX_tag_sample_buffer(context, load, userptr);
    glad_gl_load_GL_SGI_color_table(context, load, userptr);
    glad_gl_load_GL_SUNX_constant_data(context, load, userptr);
    glad_gl_load_GL_SUN_global_alpha(context, load, userptr);
    glad_gl_load_GL_SUN_mesh_array(context, load, userptr);
    glad_gl_load_GL_SUN_triangle_list(context, load, userptr);
    glad_gl_load_GL_SUN_vertex(context, load, userptr);



    return version;
}


int gladLoadGLContext(GladGLContext *context, GLADloadfunc load) {
    return gladLoadGLContextUserPtr(context, glad_gl_get_proc_from_userptr, GLAD_GNUC_EXTENSION (void*) load);
}



 


#ifdef __cplusplus
}
#endif
