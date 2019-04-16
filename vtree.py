import os
import numpy as np
import random
import scipy.spatial
import marshal
import cPickle as pickle
import time
import shutil


"""
the basic idea:
- split a node by randomly choosen points into subnodes
- attach each node to the subnode with the smallest distance
- for knn search maintain tau (the search radius) distance from query point to the farthest point of the currently found n nearest. if not n have been found, use infinity
- start with the closest node to the query point
- only need to look into nodes which are no futher than (the closest node+tau) away from query point

"""

def dist(a,b):
    #v=1-scipy.spatial.distance.cosine(a,b)
    v=np.inner(a,b)/(np.sqrt(np.sum(a*a))*np.sqrt(np.sum(b*b)))
    u=np.clip(v,-1,1)
    return np.arccos(u)

        
class VNode:
    def __init__(self,nid,tree):
        self.nid=nid
        self.tree=tree
        self.data=[]
        self.children=[]
        self._load()

    def _load(self):
        self.data,self.children=self.tree._load(self.nid)
        self._clean()

    def _save(self):
        self.tree._save(self.nid,(self.data,self.children))
        self._clean()

    def _dirty(self):
        self.tree._dirty[self.nid]=self

    def _clean(self):
        try:
            del self.tree._dirty[self.nid]
        except KeyError:
            pass

    def selectSplits(self):
        if len(self.children)==0:
            a,b=map(lambda x: x[0],random.sample(self.data,2))
            d=dist(a,b)
            for x in self.data:
                d0=dist(a,x[0])
                d1=dist(b,x[0])
                if d0>d1:
                    if d0>d:
                        d=d0
                        b=x[0]
                else:
                    if d1>d:
                        d=d1
                        a=x[0]
            self.children=[(a,self.tree._newNid()),(b,self.tree._newNid())]
            for o,nid in self.children:
                self.tree._getNode(nid)
            self._dirty()

    def _split(self,depth):
        self.selectSplits()
        for x in self.data:
            o,nid=min(self.children,key=lambda o_nid: dist(x[0],o_nid[0]))
            self.tree._getNode(nid).insert(depth+1,*x)
        del self.data[:]
        self._dirty()

    def insert(self,depth,*args):
        if len(self.children)==0:
            self.data.append(args)
            if len(self.data)>self.tree.LEAF_SIZE:
                self._split(depth)
            self._dirty()
        else:
            o,nid=min(self.children,key=lambda o_nid: dist(args[0],o_nid[0]))
            self.tree._getNode(nid).insert(depth+1,*args)

    def remove(self,k):
        if len(self.children)==0:
            for i,x in enumerate(self.data):
                if np.allclose(k,x[0]):
                    del self.data[i]
                    self._dirty()
                    return x
        else:
            o,nid=min(self.children,key=lambda o_nid: dist(args[0],o_nid[0]))
            return self.tree._getNode(nid).remove(k)

    def flatQuery(self,k):
        if len(self.children)==0:
            tmp=map(lambda x: (dist(k,x[0]),x),self.data)
            tmp.sort(key=lambda x: x[0])
            for x in tmp:
                yield x
        else:
            its=[]
            for o,nid in self.children:
                it=self.tree._getNode(nid).flatQuery(k)
                try:
                    its.append((it.next(),it))
                except StopIteration:
                    pass
            while its:
                its.sort(key=lambda x: x[0],reverse=True)
                x,it=its.pop()
                yield x
                try:
                    its.append((it.next(),it))
                except StopIteration:
                    pass
                

    def query(self,k,nids=None,lids=None):
        if nids is not None:
            nids.add(self.nid)
        if len(self.children)==0:
            if lids is not None:
                lids[self.nid]=len(self.data)
            tmp=map(lambda x: (dist(k,x[0]),x),self.data)
            tmp.sort(key=lambda x: x[0])
            for x in tmp:
                yield x
        else:
            tmp=map(lambda x: (dist(k,x[0]),x),self.children)
            tmp.sort(key=lambda x: x[0],reverse=True)
            its=[]
            while tmp or its:
                if len(its)==0:
                    d0,(o0,nid0)=tmp.pop()
                    n=self.tree._getNode(nid0)
                    it=n.query(k,nids,lids)
                    try:
                        its.append((it.next(),it,d0))
                        its.sort(key=lambda x: x[0],reverse=True)
                    except:
                        pass
                    continue
                #print "a",tmp
                #print "b",its
                if len(tmp)>0 and (its[-1][0][0]+its[-1][-1])>tmp[-1][0]:
                    d0,(o0,nid0)=tmp.pop()
                    n=self.tree._getNode(nid0)
                    it=n.query(k,nids,lids)
                    try:
                        its.append((it.next(),it,d0))
                        its.sort(key=lambda x: x[0],reverse=True)
                    except:
                        pass
                    continue
                x,it,d=its.pop()
                yield x
                try:
                    its.append((it.next(),it,d))
                    its.sort(key=lambda x: x[0],reverse=True)
                except:
                    pass
                
            
            
                
        

class VTree:
    MAX_CACHE_SIZE=8192
    LEAF_SIZE=64
    def __init__(self,path,dims):
        self.dims=dims
        self.path=os.path.abspath(path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.dpath=os.path.join(self.path,"dat")
        self.cpath=os.path.join(self.path,"cmt")
        for p in [self.dpath,self.cpath]:
            if not os.path.exists(p):
                os.makedirs(p)
        self._cache=[{},{}]
        self._dirty={}
        self._meta={"size":0,"nodes":1}
        self._finish()
        self._loadMeta()
        
    def _loadMeta(self):
        try:
            with open(os.path.join(self.dpath,"meta"),"rb") as fp:
                self._meta=pickle.load(fp)
        except:
            self._meta={"size":0,"nodes":1}

    def _finish(self):
        if os.path.exists(os.path.join(self.cpath,"commit")):
            print "commiting"
            for x in os.listdir(self.cpath):
                if x!="commit":
                    shutil.move(os.path.join(self.cpath,x),os.path.join(self.dpath,x))
        for x in os.listdir(self.cpath):
            print "removing",x
            os.remove(os.path.join(self.cpath,x))

    def commit(self):
        with open(os.path.join(self.cpath,"meta"),"wb") as fp:
            pickle.dump(self._meta,fp,protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.cpath,"commit"),"wb") as fp:
            fp.write("{}")
        self._finish()

    def flush(self):
        with open(os.path.join(self.cpath,"meta"),"wb") as fp:
            pickle.dump(self._meta,fp,protocol=pickle.HIGHEST_PROTOCOL)
        for n in self._dirty.values():
            n._save()
        

    def _getNode(self,nid):
        for c in [self._dirty]+self._cache:
            ret=c.get(nid)
            if ret is not None:
                self._cache[0][nid]=ret
                if len(self._cache[0])>self.MAX_CACHE_SIZE:
                    self._cache.insert(0,{})
                    self._cache.pop()
                return ret
        ret=VNode(nid,self)
        self._cache[0][nid]=ret
        return ret

    def _load(self,nid):
        for path in [self.cpath,self.dpath]:
            f1=os.path.join(path,nid)
            if os.path.exists(f1):
                with open(f1,"rb") as fp:
                    return pickle.load(fp)
        return [],[]

    def _save(self,nid,data):
        with open(os.path.join(self.cpath,nid),"wb") as fp:
            pickle.dump(data,fp,protocol=pickle.HIGHEST_PROTOCOL)
    
    def _newNid(self):
        while True:
            nid=str(random.random())
            if nid in self._dirty:
                continue
            if nid in self._cache[0]:
                continue
            if nid in self._cache[1]:
                continue
            fname=os.path.join(self.cpath,str(nid))
            if os.path.exists(fname):
                continue
            fname=os.path.join(self.dpath,str(nid))
            if os.path.exists(fname):
                continue
            self._meta["nodes"]+=1
            return nid
        
    def insert(self,k,*args):
        self._getNode("0").insert(0,k,*args)
        self._meta["size"]+=1

    def remove(self,k):
        ret=self._getNode("0").remove(k)
        if ret is not None:
            self._meta["size"]-=1
        return ret

    def query(self,k):
        nids=set()
        lids={}
        for x in self._getNode("0").query(k,nids,lids):
            cnt=sum(lids.values())
            yield x,len(nids),len(lids),cnt,float(cnt)/len(lids)

    def flatQuery(self,k):
        for x in self._getNode("0").flatQuery(k):
            yield x

if __name__=="__main__":
    isize=10
    t=VTree("/tmp/vt/"+str(isize),isize)
    #k=np.random.randn(isize)
    while True:
        for i in xrange(100000):
            v=np.random.randn(isize)                
            t.insert(v)
            if ((i+1)%1000)==0:
                t.flush()
                if ((i+1)%10000)==0:
                    t.commit()
                k=np.random.randn(isize)
                t0=time.time()
                for j,x in enumerate(t.query(k)):
                    print j,x,t._meta,len(t._cache[0])
                    if j>=5:
                        break
                print "i",time.time()-t0
                #if ((i+1)%10000)==0:
                #    t0=time.time()
                #    for j,(x,y) in enumerate(zip(t.flatQuery(k),t.query(k))):
                #        print j,x,y,t._meta,len(t._cache[0])
                #        if j>=5:
                #            break
                #    print "r",time.time()-t0
        t.commit()
    

        
        
        
