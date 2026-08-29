
---

## Appendix — Monitoring stack (optional segment, ~60s)

Beyond the brief, but it makes a strong visual. Add after Part 7.

```bash
kubectl get pods,svc -n monitoring
kubectl port-forward -n monitoring service/grafana 3000:3000
kubectl port-forward -n monitoring service/prometheus 9090:9090
```

**Browser → http://localhost:9090/targets**
> "Prometheus discovers targets through the Kubernetes API using endpoint
> discovery, so it scrapes each Pod individually. Both replicas, both up."

**Browser → http://localhost:3000 → Dashboards → Cats vs Dogs API**
> "Grafana's datasource and this dashboard are provisioned from ConfigMaps, not
> built by hand, so the whole stack is reproducible from the manifests alone.
> Request rate and latency are broken out per Pod - which is why this matters:
> the API's own /metrics is per-process, so each replica only knows its own
> traffic. Prometheus scraping both and summing them is what gives a complete
> picture."
